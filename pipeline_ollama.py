import argparse
from argparse import Namespace
from recipe_state_tracker import RecipeStateTracker
from utils import TEMPLATES, PROMPTS
from ollama import chat
from ollama import ChatResponse
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
    return json_objects

def query_model(model_name: str, system_prompt: str ,user_prompt: str) -> str:
    """Query the Ollama model and return the response."""
    response: ChatResponse = chat(model='llama3.2', messages=[
    {
        'role': 'system',
        'content': system_prompt,
    },{
        'role': 'user',
        'content': user_prompt,
    }])
    return response['message']['content']


def main():

    model_name = "llama3.2"
    st = RecipeStateTracker()
    # Exit the loop using CTRL+C
    while True:
        try:
            # Wait for the user input
            user_input = input("User: ")

            # Generate NLU response
            nlu_output = query_model(model_name, PROMPTS["NLU"], user_input)
            nlu_output = extract_json_from_text(nlu_output)
            print(f"NLU: {nlu_output}")
            st.update(nlu_output)
            print(f"State: {st.to_string()}")

            # Generate DM response
            dm_output = query_model(model_name, PROMPTS["DM"], st.to_string())
            print(f"DM: {dm_output}")
            dm_output = extract_json_from_text(dm_output)
            print(f"DM: {dm_output}")

            if "action" in dm_output:
                if dm_output["action"] == "confirmation_{recipe_search}" or dm_output["action"] == "confirmation_recipe_search":
                    meals = get_meals_by_ingridients(dm_output["slots"]["ingredients"])
                    print(f"Meals: {meals}")

            nlg_input = {"dm": dm_output, "nlu": st.to_dict()}
            nlg_input = json.dumps(nlg_input, indent=4)

            # Generate NLG response
            nlg_output = query_model(model_name, PROMPTS["NLG"], nlg_input)

            print(f"NLG: {nlg_output}")

        except KeyboardInterrupt:
            print("\nExiting...")
            break


if __name__ == "__main__":
    main()
