from argparse import Namespace
from typing import Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedTokenizer,
    PreTrainedModel,
)

MODELS = {
    "llama2": "meta-llama/Llama-2-7b-chat-hf",
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
}

TEMPLATES = {
    "llama2": "<s>[INST] <<SYS>>\n{}\n<</SYS>>\n\n{} [/INST]",
    "llama3": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>",

}
PROMPTS = {

    "NLU_INTENT": """
    You are the intent detection module of a recipe bot. Your task is to analyze the user input and determine the user's intent.

    ### Key Guidelines:
    1) **Intent Detection**:
    - Identify the user's intent based on their input. The possible intents are:
        - `recipe_recommendation`: The user is looking for a recipe suggestion, here the user dose not known the name of the recipe, but he would like to search a recipe.
        - `ask_for_ingredients`: The user wants to know the ingredients of a recipe.
        - `ask_for_procedure`: The user wants to know the procedure for a recipe.
        - `ask_for_time`: The user wants to know how much time is needed for a recipe.
        - `not_supported`: The user input does not match any of the above intents.

    2) **Output Format**:
    - Always return a JSON object with the following structure:
        ```json
        {
            "intent": "<detected_intent>"
        }
        ```

    ### Example:

    User Input: "Can you suggest an Italian pasta recipe?"
    Output:
    ```json
    {
        "intent": "recipe_recommendation"
    }```
    """,

    "NLU_SLOTS_recipe_recommendation": """
    You are the slot extraction module for the `recipe_recommendation` intent in a recipe bot. Your task is to extract relevant slot values from the user input.

    ### Key Guidelines:
    1) **Slot Extraction**:
    - Extract the following slots from the user input:
        - `nationality` (e.g., Italian, Tunisian, Spanish)
        - `category` (e.g., pasta, meat, vegetarian)
        - `ingredients` (a list of ingredients, e.g., tomato, onion, garlic)
    - If a slot value is not explicitly provided, set it to `null`.

    2) **Output Format**:
    - Always return a JSON object with the following structure:
        ```json
        {
            "slots": {
                "nationality": "<value_or_null>",
                "category": "<value_or_null>",
                "ingredients": "<value_or_null>"
            }
        }
        ```

    ### Example:

    User Input: "Can you suggest an Italian pasta recipe with tomato and garlic?"
    Output:
    ```json
    {
        "slots": {
            "nationality": "Italian",
            "category": "pasta",
            "ingredients": "tomato, garlic"
        }
    }```

    User Input: "I would like to cook something with chicken and potatoes."
    Output:
    ```json
    {
        "slots": {
            "nationality": null,
            "category": null,
            "ingredients": "chicken, potatoes"
        }
    }```

    """,

    "NLU_SLOTS_ask_for_ingredients": """
    You are the slot extraction module for the `ask_for_ingredients` intent in a recipe bot. Your task is to extract the `recipe_name` slot from the user input.

    ### Key Guidelines:
    1) **Slot Extraction**:
    - Extract the following slot from the user input:
        - `recipe_name` (the name of the recipe in question)
    - If the recipe name is not explicitly provided, set it to `null`.

    2) **Output Format**:
    - Always return a JSON object with the following structure:
        ```json
        {
            "slots": {
                "recipe_name": "<value_or_null>"
            }
        }
        ```

    ### Example:

    User Input: "What are the ingredients for Kedgeree?"
    Output:
    ```json
    {
        "slots": {
            "recipe_name": "Kedgeree"
        }
    }```
    """,

    "NLU_SLOTS_ask_for_procedure": """
    You are the slot extraction module for the `ask_for_procedure` intent in a recipe bot. Your task is to extract the `recipe_name` slot from the user input.

    ### Key Guidelines:
    1) **Slot Extraction**:
    - Extract the following slot from the user input:
        - `recipe_name` (the name of the recipe in question)
    - If the recipe name is not explicitly provided, set it to `null`.

    2) **Output Format**:
    - Always return a JSON object with the following structure:
        ```json
        {
            "slots": {
                "recipe_name": "<value_or_null>"
            }
        }
        ```

    ### Example:

    User Input: "How do I cook Kedgeree?"
    Output:
    ```json
    {
        "slots": {
            "recipe_name": "Kedgeree"
        }
    }```
    """,

    "DM_recipe_recommendation": """You are a dialogue manager of a recipe bot responsible for determining the `action_required` field.
    You will receive a JSON composed by the NLU component, composed by the keys `intent`, `slots`, and `sentiment` .
    - `intent` and `sentiment` are strings or null.
    - `slots` is a dictionary containing slot values.
    And also you will receive a list of recipes that match the user's request.

    Your task:
    1) Identify any null values in the `slots` dictionary.
    2) Fill the `action_required` field:
       - If the list of recipe is empty, but there are some slots with values, put inside `action_required` the action `no_recipe_found`.
       - If the list of recipe is not empty, put inside `action_required` the action `propose_recipe` and the for each null slot, put inside `action_required` the action `req_info_{slot_name}`, where `{slot_name}` is the name of a null slot.
    Return a JSON object with a single key, `action_required`, containing a list of actions to perform.

    example: 
    {
        "action_required": ["propose_recipe","req_info_category", "req_info_ingredients"]
    }
    """,

    "DM_recipe_information": """You are a dialogue manager of a recipe bot responsible for determining the `action_required` field for the intent of recipe information.
    You will receive a JSON input with keys `intent`, `slots`, and `sentiment`.
    - `intent` and `sentiment` are strings or null.
    - `slots` is a dictionary containing slot values: reicpe_name and request.
    
    Your task is populate the action_required field:
    1) If the recipe_name is not found in the database, set `action_required` to `no_recipe_found`.
    2) if the recipe_name is filled, but the request is null, set `action_required` to `req_what_info_about_recipe`.
    3) If both recipe_name and request are filled, you have to read properly the request and understand what information want the user setting `action_required` to `one of the following actions: 
        - req_ingredients: the user want to know the ingredients of the recipe
        - req_procedure: the user want to know the procedure of the recipe
        - req_time: the user want to know the time of the recipe

    Return a JSON object with a single key, `action_required`, containing the determined value.
    """,


    "NLG_recipe_recommendation": """You are a natural language generation module in a recipe bot that has to reply to the intent of recipe reccomendation.
    Based on the input, you must generate the correct request and/or reply for the user.
    You have recived the list action required from the DM module.
    The actions are:
        - no_recipe_found: The bot has not found any recipe that matches the user's request. You should tell to the user that there are no recipes that match the request. And ask to him if he want to change his request.
        - propose_recipe: The bot has found some recipes that match the user's request. You should provide the recipes to the user, the recipes are in the list of recipes that you have recived from the DM module.
        - req_info_{slot_name}: The bot needs more information about the slot_name. You should ask the user to provide more information about the slot_name if he want to filter more the recipes.
    - Example output:
      "With the information you have provided, you could cook Italian Lasagna. Do you want to know the recipe? Otherwise, please provide more details, like the ingredients you have in your fridge."
    Rembember to provide the recipe if there are some.
    **Reply only with the appropriate request or information for the user.**
    """,

    "NLG_recipe_information": """You are a natural language generation module in a recipe bot that has to reply to the intent of recipe information.
    Based on the input, you must generate the correct request and/or reply for the user.
    The user has provided a recipe, and also a request about this recipe, for example the ingredients, the procedure or something like this.
    You have recived also the action required from the DM module.
    Input:
    - A JSON object containing:
      - `NLU` dictionary with intent and slots extracted from user input.
      - `DM` dictionary with the action required and additional information.
      - The recipe name and all the information avaiable from that reciope.
    Instructions:
    - Provide the answer to the user's request which is inside the field `action_required` from the DM dictionary.
    - Example output:
        "In order to cook the lasagna you need tomato, onion, garlic, and pasta. Do you want to know how to proceed with the recipe?"
    **Reply only with the appropriate request or information for the user, and don't put things like: Here a possible response.**
    """

}



def load_model(args: Namespace) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto" if args.parallel else args.device, 
        torch_dtype=torch.float32 if args.dtype == "f32" else torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    return model, tokenizer  # type: ignore


def generate(
    model: PreTrainedModel,
    inputs: BatchEncoding,
    tokenizer: PreTrainedTokenizer,
    args: Namespace,
) -> str:
    output = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=args.max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(
        output[0][len(inputs.input_ids[0]) :], skip_special_tokens=True
    )
