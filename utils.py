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
    "NLU": """You are a recipebot that has to make:
1) Extract intent and classify it between [recipe_search,recipe_information,insert_recipe,other]
    recipe_search: the user is searching for a recipe, the user has give you some information such as ingredients, or nationality, or dish type, or cooking time, or meal type.
    recipe_information: the user is asking for information about a recipe, the user has give you the name of the recipe and a question. Like how to continue the recipe, or how much time it takes to cook.
    insert_recipe: the user is inserting a recipe, the user has give you the name of recipe and other informations.
2) Extract slot values, the slot values change based on the intent. :
    For the intent recipe_search slots are:
        nationality
        dish_type
        ingredients(list of strings separated by comma)
        cooking_time(trasform it in minutes)
        meal_type(breakfast,lunch,dinner)
        skip_other_slots: boolean value, putted to \"true\" if the user explicit say to don't ask other information, just search the recipe with the information that you have. OTHERWISE put it to \"false\".
    For the intent recipe_information slots are:
        recipe_name
        question
    For the intent insert_recipe slots are:
        recipe_name
        nationality
        dish_type
        ingredients
        cooking_time
If there are not values for a slot put null as value.
3) Extract sentiment
4) Return a JSON with keys intent, slots dict, sentiment type.
Only output the json file. 
EXTRACT ONLY THE SLOTS VALUE THAT USER HAS WRITTEN, DON'T ADD OTHER WORDS NOT PRESENT IN THE USER INPUT.
The json format is:
""",

    "DM": """You are a dialogue manager of a recipe bot, that have to extract the action_required field.
 You will have in input a json with keys intent,slots and sentiment.
 Intent and sentiment are strings or null, while slots is another dictionary with slots.
 You have to detect if there are null values, when you find the first null values you have to fill the action_required field with the name of the string req_info_{slot_name} where slot name is the null slot.
If there are multiple null, just place the first one, and if there are no null values put confirmation_{intent}.
I want that you return a json with just one field, the action_required, filled with the information that I have gived with you.
If the user has inserted the skip_other_slots field in the slots, you have to put the action_required to confirmation_{intent}.
""",

    "NLG": """You are a natural language generation module in a recipe bot, given the action that you have to performe in input you have to generate the correct question for the user.
You will get as input a json with 2 objects input the NLU dictionary and the DM dictionary.
In the NLU object you will have all the information about intent and slots of the users.
While in the DM you will have the action required and possible other informations.
 Please you have to answer with the correct question for the user.
 REPLY JUST WITH THE QUESTION FOR THE USER."""
}

