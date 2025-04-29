import sys
from pathlib import Path

# Add project root to sys.path
root = Path(__file__).resolve().parent
sys.path.append(str(root))

import torch
import gc
import json

from config import (
    FINETUNED_MODEL_DIR,
    MAX_INPUT_LENGTH,
    MAX_OUTPUT_LENGTH,
)
from utils.model_utils import load_model, load_tokenizer
from utils.data_loader import load_and_split_dataset
from utils.data_preparation import preprocess
from utils.evaluation import evaluate_in_chunks
from utils.gpu_logger import log_gpu
from pretokenizers.firstpretokenizer import FirstPretokenizer
from config import GENERATION_ARGS, CHUNK_SIZE, SAVE_OUTPUTS_PATH, NUM_EXAMPLES_TO_GENERATE, GENERATED_OUTPUTS_DIR
from utils.metrics import compute_metrics, save_metrics_to_file
import os


def main():
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load fine-tuned model and tokenizer
    model = load_model(FINETUNED_MODEL_DIR).to(device)
    tokenizer = load_tokenizer(FINETUNED_MODEL_DIR)

    pretokenizer = FirstPretokenizer(_use_dedent=True, _use_semantics=True)

    # Load test dataset
    dataset_dict = load_and_split_dataset()
    raw_test_set = dataset_dict["test"].select(range(NUM_EXAMPLES_TO_GENERATE))

    # Preprocess test dataset
    tokenized_test_set = raw_test_set.map(
        lambda batch: preprocess(batch, tokenizer, MAX_INPUT_LENGTH, MAX_OUTPUT_LENGTH),
        batched=True,
        remove_columns=raw_test_set.column_names
    )

    os.makedirs(GENERATED_OUTPUTS_DIR, exist_ok=True)

    # === Generate outputs ===
    print("\n=== Generating outputs with fine-tuned model ===")
    outputs = evaluate_in_chunks(
        model=model,
        dataset=tokenized_test_set,
        chunk_size=CHUNK_SIZE,
        save_outputs_path=SAVE_OUTPUTS_PATH,
        tokenizer=tokenizer,
        raw_dataset=raw_test_set,
        pretokenizer=pretokenizer,
        max_input_length=MAX_INPUT_LENGTH,
        generation_args=GENERATION_ARGS,
    )

    # === Compute real metrics ===
    # Prepare predictions and references
    predictions = [sample["prediction"] for sample in outputs]
    references = [sample["reference"] for sample in outputs]

    # compute_metrics expects the same structure as trainer.evaluate usually gives
    # We'll adapt it manually
    eval_pred = (predictions, references)

    metrics = compute_metrics(eval_pred, tokenizer=None, pretokenizer=None)

    # Save metrics to file
    metrics_save_path = os.path.join(GENERATED_OUTPUTS_DIR, "generated_metrics.json")
    save_metrics_to_file(metrics, metrics_save_path)

    print(f"✅ Saved evaluation metrics to {metrics_save_path}")

    # Cleanup
    torch.cuda.empty_cache()
    gc.collect()
    log_gpu()



if __name__ == "__main__":
    main()