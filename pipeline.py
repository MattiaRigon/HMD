import argparse
from argparse import Namespace
from recipe_state_tracker import RecipeStateTracker
import torch
from utils_cluster import load_model, generate, MODELS, TEMPLATES, PROMPTS
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
    st = RecipeStateTracker()

    # exit the loop using CTRL+C
    historical_context = []
    while True:
        # function to wait for the user input
        user_input = input("User: ")
        historical_context = historical_context[:4]

        nlu_input= {"user_input": user_input, "historical_context": historical_context}
        historical_context.append(user_input)
        # get the NLU output
        nlu_text = args.chat_template.format(PROMPTS["NLU"], user_input)
        nlu_input = tokenizer(nlu_text, return_tensors="pt").to(model.device)
        nlu_output = generate(model, nlu_input, tokenizer, args)
        # print("NLU Output: ", nlu_output)
        nlu_output = extract_json_from_text(nlu_output)
        print(f"NLU: {nlu_output}")
        st.update(nlu_output)
        # print(f"State: {st.to_string()}")

        # get information from the database
        if nlu_output["intent"] == "recipe_recommendation":
            recipe_recommendation_slots = st.get_slots("recipe_recommendation")
            filtered_recipes = filter_recipes(recipe_recommendation_slots["nationality"], recipe_recommendation_slots["category"], recipe_recommendation_slots["ingredients"])
            print(f"Meals: {filtered_recipes}")

            dm_input = {"matched_recipes": filtered_recipes, "state": st.to_dict()}
        
        elif nlu_output["intent"] == "ask_for_ingredients" or nlu_output["intent"] == "ask_for_procedure" or nlu_output["intent"] == "ask_for_time":
            recipe_information_slots = st.get_slots(nlu_output["intent"])
            recipe_information = get_meal_by_name(recipe_information_slots["recipe_name"])

            dm_input = {"recipe": recipe_information, "state": st.to_dict()}

            # print(f"RECIPE INFORMATION: {recipe_information}")
        else:

            dm_input = {"state": st.to_dict()}
            recipe_information = []
            recipe_information_slots = []
        

        # get the DM output
        dm_input = json.dumps(dm_input, indent=4)
        
        if nlu_output["intent"] == "recipe_recommendation" or nlu_output["intent"] == "not_supported":

            dm_text = args.chat_template.format(PROMPTS[f"DM_{nlu_output['intent']}"], dm_input)
            dm_input = tokenizer(dm_text, return_tensors="pt").to(model.device)
            dm_output = generate(model, dm_input, tokenizer, args)
            dm_output = extract_json_from_text(dm_output)
        elif nlu_output["intent"] == "ask_for_ingredients":
            dm_output = {"action_required": ["provide list of ingredients"]}
            dm_output = json.dumps(dm_output, indent=4)
        elif nlu_output["intent"] == "ask_for_procedure":
            dm_output = {"action_required": ["provide procedure of the recipe"]}
            dm_output = json.dumps(dm_output, indent=4)
        elif nlu_output["intent"] == "ask_for_time":
            dm_output = {"action_required": ["provide the time needed for the recipe"]}
            dm_output = json.dumps(dm_output, indent=4)
            
        # print(f"DM: {dm_output}")

        # Optional Pre-Processing for NLG
        if nlu_output["intent"] == "recipe_recommendation":
            nlg_input = {"dm": dm_output, "nlu": st.to_dict(), "recipes": filtered_recipes}
            prompt = PROMPTS[f"NLG_{nlu_output['intent']}"]
        elif nlu_output["intent"] == "ask_for_ingredients" or nlu_output["intent"] == "ask_for_procedure" or nlu_output["intent"] == "ask_for_time":
            nlg_input = {"dm": dm_output, "nlu": st.to_dict(), "recipe": recipe_information}
            prompt = PROMPTS[f"NLG_recipe_information"]
        nlg_input = json.dumps(nlg_input, indent=4)

        # print(f"NLG Input: {nlg_input}")
        # get the NLG output
        print("NLg Input: ", nlg_input)
        nlg_text = args.chat_template.format(prompt, nlg_input)
        nlg_input = tokenizer(nlg_text, return_tensors="pt").to(model.device)
        nlg_output = generate(model, nlg_input, tokenizer, args)

        print(f"NLG: {nlg_output}")
        historical_context.append(nlg_output)


if __name__ == "__main__":
    main()
