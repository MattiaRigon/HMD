import argparse
from argparse import Namespace
from recipe_state_tracker import RecipeStateTracker
import torch
from utils import load_model, generate, MODELS, TEMPLATES, PROMPTS
import json
import re
from data.database import filter_recipes, get_meal_by_name, get_meals_by_ingredients
from copy import deepcopy

def extract_json_from_text(content):

    json_objects = []

    # Regular expression to match JSON-like structures
    json_pattern = re.compile(r'\{(?:[^{}]*|(?:\{[^{}]*\}))*\}', re.DOTALL)

    # Find all JSON matches in the content
    matches = json_pattern.findall(content)

    for match in matches:
        try:
            # Parse the JSON string to ensure it is valid
            json_obj = json.loads(match)
            json_objects.append(json_obj)
        except json.JSONDecodeError:
            print(f"Invalid JSON detected and skipped: {match}...")
    if len(json_objects) == 0:
        return {}
    else:
        return json_objects[0]

def get_args() -> Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m query_model",
        description="Query a specific model with a given input.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "model_name",
        type=str,
        choices=list(MODELS.keys()),
        help="The model to query.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="The device to use for the model.",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Split the model across multiple devices.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["f32", "bf16"],
        default="bf16",
        help="The data type to use for the model.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1000,
        help="The maximum sequence length to use for the model.",
    )

    parsed_args = parser.parse_args()
    parsed_args.chat_template = TEMPLATES[parsed_args.model_name]
    parsed_args.model_name = MODELS[parsed_args.model_name]

    return parsed_args
def main():
    args = get_args()
    model, tokenizer = load_model(args)
    state_tracker = RecipeStateTracker()

    historical_context = []
    while True:
        user_input = get_user_input(historical_context)
        intents = process_nlu(user_input, state_tracker, historical_context, model, tokenizer, args)
        nlgs = []
        for intent in intents:
            nlu = {"intent": intent, "slots": {}}
            nlus = [nlu]
            if intent!= "not_supported":
                update_nlu_slots(nlu, user_input, state_tracker, model, tokenizer, args)  
                if nlu["intent"] == "recipe_recommendation":
                    if isinstance(nlu["slots"]["nationality"], str):
                        nlu["slots"]["nationality"] =  nlu["slots"]["nationality"].replace(" ","").split(",")
                    if isinstance(nlu["slots"]["nationality"], list):
                        nationalities = nlu["slots"]["nationality"]
                        nlu["slots"]["nationality"] = nationalities[0]
                        nationalities = nationalities[1:]
                        for i in range(len(nationalities)):
                            new_nlu = deepcopy(nlu)
                            new_nlu["slots"]["nationality"] = nationalities[i]
                            nlus.append(new_nlu)
                if nlu["intent"] in ["ask_for_ingredients","ask_for_procedure","ask_for_time"]:
                    if isinstance(nlu["slots"]["recipe_name"], str):
                        nlu["slots"]["recipe_name"] =  nlu["slots"]["recipe_name"].replace(" ","").split(",")
                    if isinstance(nlu["slots"]["recipe_name"], list):
                        recipe_names = nlu["slots"]["recipe_name"]
                        if len(recipe_names) == 0:
                            nlu["slots"]["recipe_name"] = None
                        else:
                            nlu["slots"]["recipe_name"] = recipe_names[0]
                            recipe_names = recipe_names[1:]
                            for i in range(len(recipe_names)):
                                new_nlu = deepcopy(nlu)
                                new_nlu["slots"]["recipe_name"] = recipe_names[i]
                                nlus.append(new_nlu)
            for nlu in nlus:
                state_tracker.update(nlu)
                
                dm_input, filtered_recipes, recipe_information = generate_dm_input(nlu, state_tracker)
                dm_output = generate_dm_output(nlu, dm_input, filtered_recipes, recipe_information, model, tokenizer, args)
                
                nlg_input, prompt = prepare_nlg_input(nlu, state_tracker, dm_output, filtered_recipes, recipe_information)
                nlg_output = generate_nlg_output(nlg_input, prompt, model, tokenizer, args)
                nlgs.append(nlg_output)                

        if len(nlgs) > 1:
            nlg_input = nlgs
            prompt = PROMPTS["NLG_END"]
            nlg_output = generate_nlg_output(nlg_input, prompt, model, tokenizer, args)
            print(f"NLG OUTPUT: {nlg_output}")
            historical_context.append(nlg_output)
        else:
            print(f"NLG OUTPUT: {nlgs[0]}")
            historical_context.append(nlgs[0])



def get_user_input(historical_context):
    user_input = input("User: ")
    historical_context.append(user_input)
    historical_context = historical_context[:3]
    return user_input


def process_nlu(user_input, state_tracker, historical_context, model, tokenizer, args):
    if len(historical_context) >= 2:
        context = historical_context[-2]
    else:
        context = ""
    nlu_input = {"user_input": user_input,"historical_context": context}
    print(f"NLU Input: {nlu_input}")

    nlu_text = args.chat_template.format(PROMPTS["NLU_INTENT"], nlu_input)
    tokenized_input = tokenizer(nlu_text, return_tensors="pt").to(model.device)
    nlu_output = generate(model, tokenized_input, tokenizer, args)
    nlu_output = extract_json_from_text(nlu_output)
    print(f"NLU INTENT: {nlu_output}")
    if "intents" not in list(nlu_output.keys()):
        return ["not_supported"]
    return nlu_output["intents"]


def update_nlu_slots(nlu, user_input, state_tracker, model, tokenizer, args):
    nlu_input = {"user_input": user_input, "state_tracker": state_tracker.to_string()}
    nlu_text = args.chat_template.format(PROMPTS[f"NLU_SLOTS_{nlu['intent']}"], nlu_input)
    tokenized_input = tokenizer(nlu_text, return_tensors="pt").to(model.device)
    nlu_output = generate(model, tokenized_input, tokenizer, args)
    nlu_output = extract_json_from_text(nlu_output)
    print(f"NLU SLOTS: {nlu_output}")
    nlu["slots"] = nlu_output["slots"]
    


def generate_dm_input(nlu, state_tracker):
    filtered_recipes = []
    recipe_information = []

    if nlu["intent"] == "recipe_recommendation":
        slots = state_tracker.get_slots("recipe_recommendation")
        print(f"Slots: {slots}")
        filtered_recipes = filter_recipes(slots.get("nationality"), slots.get("category"), slots.get("ingredients"))
        print(f"Meals: {filtered_recipes}")
        return {"matched_recipes": filtered_recipes, "state": state_tracker.to_dict()}, filtered_recipes, []

    elif nlu["intent"] in {"ask_for_ingredients", "ask_for_procedure", "ask_for_time"}:
        slots = state_tracker.get_slots(nlu["intent"])
        if slots["recipe_name"] is None:
            recipe_information = None
        else:
            recipe_information = get_meal_by_name(slots["recipe_name"])
        return {"recipe": recipe_information, "state": state_tracker.to_dict()}, [], recipe_information

    return {"state": state_tracker.to_dict()}, [], []


def generate_dm_output(nlu, dm_input, filtered_recipes, recipe_information, model, tokenizer, args):
    if nlu["intent"] in {"recipe_recommendation", "not_supported"}:
        dm_text = args.chat_template.format(PROMPTS[f"DM_{nlu['intent']}"], json.dumps(dm_input, indent=4))
        tokenized_input = tokenizer(dm_text, return_tensors="pt").to(model.device)
        dm_output = generate(model, tokenized_input, tokenizer, args)
        return extract_json_from_text(dm_output)

    elif nlu["intent"] == "ask_for_ingredients":
        if recipe_information is None:
            return json.dumps({"action_required": ["ask to the user to provide recipe name for which wants the igredients"]}, indent=4)
        else:
            return json.dumps({"action_required": ["provide list of ingredients"]}, indent=4)

    elif nlu["intent"] == "ask_for_procedure":
        if recipe_information is None:
            return json.dumps({"action_required": ["ask to the user to provide recipe name for which wants know the procedure"]}, indent=4)
        else:
            return json.dumps({"action_required": ["provide procedure of the recipe"]}, indent=4)

    elif nlu["intent"] == "ask_for_time":
        if recipe_information is None:
            return json.dumps({"action_required": ["ask to the user to provide recipe name for which wants know how much time is needed in order to do the recipe"]}, indent=4)
        else:
            return json.dumps({"action_required": ["provide the time needed for the recipe"]}, indent=4)
    
    elif nlu["intent"] == "not_supported":
        return json.dumps({"action_required": ["tell to the user that the bot cannot help for his request"]}, indent=4)


def prepare_nlg_input(nlu, state_tracker, dm_output, filtered_recipes, recipe_information):
    if nlu["intent"] == "recipe_recommendation":
        return {"dm": dm_output, "nlu": state_tracker.to_dict(), "recipes": filtered_recipes}, PROMPTS[f"NLG_{nlu['intent']}"]

    elif nlu["intent"] in {"ask_for_ingredients", "ask_for_procedure", "ask_for_time"}:
        return {"dm": dm_output, "nlu": state_tracker.to_dict(), "recipe": recipe_information}, PROMPTS[f"NLG_recipe_information"]

    elif nlu["intent"] == "not_supported":
        return {"dm": dm_output, "nlu": state_tracker.to_dict()}, PROMPTS[f"NLG_not_supported"]

    raise ValueError("Invalid intent detected.")


def generate_nlg_output(nlg_input, prompt, model, tokenizer, args):
    nlg_text = args.chat_template.format(prompt, json.dumps(nlg_input, indent=4))
    tokenized_input = tokenizer(nlg_text, return_tensors="pt").to(model.device)
    return generate(model, tokenized_input, tokenizer, args)

if __name__ == "__main__":
    main()
