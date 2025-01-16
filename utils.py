from argparse import Namespace
from typing import Tuple

MODELS = {
    "llama2": "meta-llama/Llama-2-7b-chat-hf",
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
}

TEMPLATES = {
    "llama2": "<s>[INST] <<SYS>>\n{}\n<</SYS>>\n\n{} [/INST]",
}

PROMPTS = {

    "NLU_INTENT": """You are a natural language understanding module of a recipe bot that has to extract the intent of the user input.:
    1) Extract intent and classify it between [recipe_recommendation,recipe_information,insert_recipe,other]
    recipe_recommendation: the user is searching for a recipe, the user has give you some information such as ingredients, or nationality, or dish type, or cooking time, or meal type.
    recipe_information: the user is asking for information about a recipe, the user has give you the name of the recipe and a request. Like how to continue the recipe, or how much time it takes to cook.
    """,
    "NLU":""""
    You are a natural language understanding module of a recipe bot, that has to extract slots and sentiment of the user input.
    1) Extract slot values, the slot values change based on the intent. :
    For the intent recipe_recommendation slots are:
        nationality (like italian, tunisian, spanish)
        category (like pasta, meat, fish)
        ingredients (like tomato, onion, garlic)
    For the intent recipe_information slots are:
        recipe_name
        request
If there are not values for a slot put null as value.
3) Extract sentiment
4) Return a JSON with keys intent, slots dict, sentiment type.
Only output the json file. 
EXTRACT ONLY THE SLOTS VALUE THAT USER HAS WRITTEN, DON'T ADD OTHER WORDS NOT PRESENT IN THE USER INPUT.
The json format is:
""",

    "DM": """You are a dialogue manager of a recipe bot, that have to extract the action_required field.
 You will have in input a json with keys intent, slots and sentiment.
 Intent and sentiment are strings or null, while slots is another dictionary with slots.
 You have to detect if there are null values, when you find the first null values you have to fill the action_required field with the name of the string req_info_{slot_name} where slot name is the null slot.
If there are multiple null, just place the first one, and if there are no null values put confirmation_{intent}.
I want that you return a json with just one field, the action_required, filled with the information that I have gived with you.
""",

    "NLG": """You are a natural language generation module in a recipe bot, given the action that you have to performe in input you have to generate the correct request for the user.
You will get as input a json with 2 objects input the NLU dictionary and the DM dictionary.
In the NLU object you will have all the information about intent and slots of the users.
While in the DM you will have the action required and possible other informations.
 Please you have to answer with the correct request for the user.
 REPLY JUST WITH THE request FOR THE USER."""
}

