import os
import sys

from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root))

from training.training_additions import T5WithModeLoss

def load_tokenizer(model_path_or_name):
    """
    Load a tokenizer from a model name (e.g., "t5-base") or a directory path (e.g., "outputs/finetuned_model").
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)
    return tokenizer


def load_model(model_path_or_name, run_custon_loss):
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

    Args:
        model: Huggingface model (e.g., T5, BART, etc.)
        tokenizer: Huggingface tokenizer
        output_dir: Path to the directory where model and tokenizer should be saved
    """
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Model and tokenizer saved successfully to '{output_dir}'")
