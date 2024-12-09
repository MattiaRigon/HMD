import argparse
from argparse import Namespace
from recipe_state_tracker import RecipeStateTracker
import torch
from utils_cluster import load_model, generate, MODELS, TEMPLATES, PROMPTS
import json
import re
from data.database import *


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
            print(f"Invalid JSON detected and skipped: {match[:30]}...")
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
        default=128,
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
    while True:
        # function to wait for the user input
        user_input = input("User: ")

        # get the NLU output
        nlu_text = args.chat_template.format(PROMPTS["NLU"], user_input)
        nlu_input = tokenizer(nlu_text, return_tensors="pt").to(model.device)
        nlu_output = generate(model, nlu_input, tokenizer, args)
        nlu_output = extract_json_from_text(nlu_output)
        print(f"NLU: {nlu_output}")
        st.update(nlu_output)
        print(f"State: {st.to_string()}")
        # Optional Pre-Processing for DM
        nlu_output = nlu_output.strip()

        # get the DM output
        dm_text = args.chat_template.format(PROMPTS["DM"], st.to_string())
        dm_input = tokenizer(dm_text, return_tensors="pt").to(model.device)
        dm_output = generate(model, dm_input, tokenizer, args)
        dm_output = extract_json_from_text(dm_output)
        print(f"DM: {dm_output}")

        if "action" in dm_output:
            if dm_output["action"] == "confirmation_{recipe_search}" or dm_output["action"] == "confirmation_recipe_search":
                meals = get_meals_by_ingridients(dm_output["slots"]["ingredients"])
                print(f"Meals: {meals}")

        # Optional Pre-Processing for NLG
        nlg_input = {"dm": dm_output, "nlu": st.to_dict()}
        nlg_input = json.dumps(nlg_input, indent=4)


        # get the NLG output
        nlg_text = args.chat_template.format(PROMPTS["NLG"], nlg_input)
        nlg_input = tokenizer(nlg_text, return_tensors="pt").to(model.device)
        nlg_output = generate(model, nlg_input, tokenizer, args)

        print(f"NLG: {nlg_output}")


if __name__ == "__main__":
    main()
