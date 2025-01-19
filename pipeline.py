import argparse
from argparse import Namespace
from recipe_state_tracker import RecipeStateTracker
import torch
from utils import load_model, generate, MODELS, TEMPLATES, PROMPTS
import json
import re
from data.database import filter_recipes, get_meal_by_name, get_meals_by_ingredients

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
        nlu = process_nlu(user_input, state_tracker, model, tokenizer, args)

        if nlu["intent"] != "not_supported":
            update_nlu_slots(nlu, user_input, state_tracker, model, tokenizer, args)

        state_tracker.update(nlu)
        
        dm_input, filtered_recipes, recipe_information = generate_dm_input(nlu, state_tracker)
        dm_output = generate_dm_output(nlu, dm_input, filtered_recipes, recipe_information, model, tokenizer, args)
        
        nlg_input, prompt = prepare_nlg_input(nlu, state_tracker, dm_output, filtered_recipes, recipe_information)
        nlg_output = generate_nlg_output(nlg_input, prompt, model, tokenizer, args)

        print(f"NLG: {nlg_output}")
        historical_context.append(nlg_output)


def get_user_input(historical_context):
    user_input = input("User: ")
    historical_context = historical_context[:4]
    return user_input


def process_nlu(user_input, state_tracker, model, tokenizer, args):
    nlu_input = {"user_input": user_input, "state_tracker": state_tracker.to_string()}
    print(f"NLU Input: {nlu_input}")

    nlu_text = args.chat_template.format(PROMPTS["NLU_INTENT"], nlu_input)
    tokenized_input = tokenizer(nlu_text, return_tensors="pt").to(model.device)
    nlu_output = generate(model, tokenized_input, tokenizer, args)
    nlu_output = extract_json_from_text(nlu_output)
    print(f"NLU INTENT: {nlu_output}")

    return {"intent": nlu_output["intent"], "slots": {}}


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
        return json.dumps({"action_required": ["provide list of ingredients"]}, indent=4)

    elif nlu["intent"] == "ask_for_procedure":
        return json.dumps({"action_required": ["provide procedure of the recipe"]}, indent=4)

    elif nlu["intent"] == "ask_for_time":
        return json.dumps({"action_required": ["provide the time needed for the recipe"]}, indent=4)


def prepare_nlg_input(nlu, state_tracker, dm_output, filtered_recipes, recipe_information):
    if nlu["intent"] == "recipe_recommendation":
        return {"dm": dm_output, "nlu": state_tracker.to_dict(), "recipes": filtered_recipes}, PROMPTS[f"NLG_{nlu['intent']}"]

    elif nlu["intent"] in {"ask_for_ingredients", "ask_for_procedure", "ask_for_time"}:
        return {"dm": dm_output, "nlu": state_tracker.to_dict(), "recipe": recipe_information}, PROMPTS[f"NLG_recipe_information"]

    raise ValueError("Invalid intent detected.")


def generate_nlg_output(nlg_input, prompt, model, tokenizer, args):
    nlg_text = args.chat_template.format(prompt, json.dumps(nlg_input, indent=4))
    tokenized_input = tokenizer(nlg_text, return_tensors="pt").to(model.device)
    return generate(model, tokenized_input, tokenizer, args)

if __name__ == "__main__":
    main()
