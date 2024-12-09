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
}

PROMPTS = {
    "NLU": """Identify the user intent from this list:
[pizza_ordering, pizza_delivery, drink_ordering, out_of_domain].
If the intent is pizza_ordering, extract the following slot values from the user input
- pizza_size, the size of the pizza
- pizza_type, the type of pizza
- pizza_count, the number of pizzas.
If no values are present in the user input you have to put null as the value.
Output them in a json format.
Only output the json file.
The json format is:
{
    "intent": "intent_value",
    "slots": {
        "slot1": "value1",
        "slot2": "value2",
        "slot3": "value3"
    }
}""",

    "DM": """You are the Dialogue Manager.
Given the output of the NLU component, you should only generate the next best action from this list:
- request_info(slot), if a slot value is missing (null) by substituting slot with the missing slot name
- confirmation(intent), if all slots have been filled""",

    "NLG": """You are the NLG component: you must be very polite.
Given the next best action classified by the Dialogue Manager (DM),
you should only generate a lexicalized response for the user.
Possible next best actions are:
- request_info(slot): generate an appropriate question to ask the user for the missing slot value
- confirmation(intent): generate an appropriate confirmation message for the user intent"""
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
