import os
import sys

from pathlib import Path
from typing import Union
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))

from data_processing.pretokenizers.firstpretokenizer import FirstPretokenizer
from model_operations.training.training_additions import T5WithModeLoss
from transformers import AutoModelForCausalLM
from config import USE_CUSTOM_EOS, EOS

def load_tokenizer(model_path_or_name: str, use_custom_eos: bool = USE_CUSTOM_EOS, pretokenizer: Union[FirstPretokenizer, None] = None):
    """
    Load a tokenizer from a model name (e.g., "t5-base") or a directory path (e.g., "outputs/finetuned_model").
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)

    if use_custom_eos:
        tokenizer.add_special_tokens({'eos_token': EOS})
        tokenizer.eos_token = EOS

    if pretokenizer:
        tags = [v for k, v in pretokenizer.tags.__dict__.items() if not k.startswith("_")]
        semantic_token_ids = [i for i in range(tokenizer.vocab_size) if i not in tokenizer.all_special_ids]

        # Adds special tokens to the tokenizer
        if not Path(model_path_or_name).exists(): # model from a hub - not memory (outputs)
            tokenizer.add_tokens(tags)

        code_token_ids = [tokenizer.convert_tokens_to_ids(tok) for tok in tags]
        semantic_start_id = tokenizer.convert_tokens_to_ids(pretokenizer.tags.SEMANTIC_START)
        semantic_end_id = tokenizer.convert_tokens_to_ids(pretokenizer.tags.SEMANTIC_END)
        semantic_token_ids.append(semantic_end_id)
        code_token_ids.remove(semantic_end_id)
        code_token_ids.append(tokenizer.convert_tokens_to_ids("</s>"))
        code_token_ids.append(tokenizer.convert_tokens_to_ids("<pad>")) 
        if use_custom_eos:
            code_token_ids.append(tokenizer.convert_tokens_to_ids(EOS))

        return tokenizer, (semantic_start_id, semantic_end_id, code_token_ids, semantic_token_ids)

    return tokenizer, None


def load_model(model_path_or_name: str, run_custom_loss: bool = False):
    """
    Load a model from a model name (e.g., "t5-base", "gpt2-xl") or a directory path.

    :param model_path_or_name: Model name or path to the model directory.
    :param run_custom_loss: If True and the model is T5-based, loads T5WithModeLoss instead of the standard model.
    :return: Loaded model instance.
    """
    # Dynamically select the model type based on the name
    print(model_path_or_name)

    if "t5" in model_path_or_name.lower():
        if run_custom_loss:
            print(f"Loading custom T5WithModeLoss for {model_path_or_name}")
            return T5WithModeLoss.from_pretrained(model_path_or_name)
        else:
            print(f"Loading standard T5 model for {model_path_or_name}")
            return AutoModelForSeq2SeqLM.from_pretrained(model_path_or_name)
    elif model_path_or_name.lower().startswith("gpt2"):
        print(f"Loading GPT2LMHeadModel for {model_path_or_name}")
        from transformers import GPT2LMHeadModel
        return GPT2LMHeadModel.from_pretrained(model_path_or_name)
    else:
        raise ValueError(f"Unsupported model type: {model_path_or_name}")

def save_model(model, tokenizer, output_dir):
    """
    Save the model and tokenizer to the specified directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Model and tokenizer saved successfully to '{output_dir}'")
