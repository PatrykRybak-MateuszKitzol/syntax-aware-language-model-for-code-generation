import os
import sys

from pathlib import Path
from typing import Union
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))

from data_processing.pretokenizers.firstpretokenizer import FirstPretokenizer
from model_operations.training.training_additions import T5WithModeLoss

def load_tokenizer(model_path_or_name: str, pretokenizer: Union[FirstPretokenizer, None] = None):
    """
    Load a tokenizer from a model name (e.g., "t5-base") or a directory path (e.g., "outputs/finetuned_model").
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)

    if Path(model_path_or_name).exists() and pretokenizer:
        semantic_token_ids = [i for i in range(tokenizer.vocab_size) if i not in tokenizer.all_special_ids]
        tags = [v for k, v in pretokenizer.tags.__dict__.items() if not k.startswith("_")]
        tokenizer.add_tokens(tags)
        code_token_ids = [tokenizer.convert_tokens_to_ids(tok) for tok in tags]
        semantic_start_id = tokenizer.convert_tokens_to_ids(pretokenizer.tags.SEMANTIC_START)
        semantic_end_id = tokenizer.convert_tokens_to_ids(pretokenizer.tags.SEMANTIC_END)
        semantic_token_ids.append(semantic_end_id)
        code_token_ids.remove(semantic_end_id)
        code_token_ids.append(tokenizer.convert_tokens_to_ids("</s>"))
        code_token_ids.append(tokenizer.convert_tokens_to_ids("<pad>")) 

    return tokenizer, (semantic_start_id, semantic_end_id, code_token_ids, semantic_token_ids)

def load_model(model_path_or_name: str, run_custon_loss: bool):
    """
    Load a model from a model name (e.g., "t5-base") or a directory path (e.g., "outputs/finetuned_model").
    """
    if run_custon_loss:
        return T5WithModeLoss.from_pretrained(model_path_or_name)
    else:
        return AutoModelForSeq2SeqLM.from_pretrained(model_path_or_name)

def save_model(model, tokenizer, output_dir):
    """
    Save the model and tokenizer to the specified directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Model and tokenizer saved successfully to '{output_dir}'")
