"""
Code to generate synthetic data to enrich the lexicon.
"""

import argparse
import torch
from unsloth import FastLanguageModel

from utils.prompts import prepare_input_texts

def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unsloth_model_id", type=str, required=True, help="The ID of the Unsloth model to use for generation.")
    parser.add_argument("--prompt_template", type=str, default="prompt1", choices=["prompt1", "prompt2"])

    # Generation Arguments
    parser.add_argument("--max_new_tokens", type=int, default=50, help="The maximum number of new tokens to generate.")
    
    args = parser.parse_args()
    return args

def main():
    args = parse_command_line_args()

    ############### Load data ################

    ############### Prepare input texts and dataloaders ###############
    # Prepare input texts

    # Prepare message format 

    # Prepare dataloader

    ############### Load model and tokenizer ###############
    model, tokenizer = FastLanguageModel.from_pretrained(args.unsloth_model_id,
                                                         device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                                                         load_in_4bit = True if 'bnb' in args.unsloth_model_id else False)
    FastLanguageModel.for_inference(model)

    ############### Generate synthetic data ###############

    ############### Post-process and save generated data ###############



if __name__ == "__main__":
    main()