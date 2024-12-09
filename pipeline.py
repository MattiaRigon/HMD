import argparse
from argparse import Namespace

import torch

from utils import load_model, generate, MODELS, TEMPLATES, PROMPTS


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

    # exit the loop using CTRL+C
    while True:
        # function to wait for the user input
        user_input = input("User: ")

        # get the NLU output
        nlu_text = args.chat_template.format(PROMPTS["NLU"], user_input)
        nlu_input = tokenizer(nlu_text, return_tensors="pt").to(model.device)
        nlu_output = generate(model, nlu_input, tokenizer, args)
        print(f"NLU: {nlu_output}")

        # Optional Pre-Processing for DM
        nlu_output = nlu_output.strip()

        # get the DM output
        dm_text = args.chat_template.format(PROMPTS["DM"], nlu_output)
        dm_input = tokenizer(dm_text, return_tensors="pt").to(model.device)
        dm_output = generate(model, dm_input, tokenizer, args)
        print(f"DM: {dm_output}")

        # Optional Pre-Processing for NLG
        dm_output = dm_output.strip()

        # get the NLG output
        nlg_text = args.chat_template.format(PROMPTS["NLG"], dm_output)
        nlg_input = tokenizer(nlg_text, return_tensors="pt").to(model.device)
        nlg_output = generate(model, nlg_input, tokenizer, args)

        print(f"NLG: {nlg_output}")


if __name__ == "__main__":
    main()
