import sys
import torch
import os
import gc

from pathlib import Path
from transformers import LogitsProcessorList

root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root))

from config import (
    FINETUNED_MODEL_DIR,
    MAX_INPUT_LENGTH,
    MAX_OUTPUT_LENGTH,
    RUN_CUSTOM_LOSS,
    RUN_LOGITS_PROCESSOR,
    GENERATION_ARGS,
    CHUNK_SIZE,
    SAVE_OUTPUTS_PATH,
    NUM_EXAMPLES_TO_GENERATE,
    GENERATED_OUTPUTS_DIR,
    TEST_SPLIT_DIR,
    FINETUNING,
    MODEL_NAME
)

from model_operations.generate_evaluate.evaluation import evaluate_in_chunks
from model_operations.generate_evaluate.metrics import compute_metrics, save_metrics_to_file
from model_operations.training.training_additions import SemanticCodeLogitsMask
from model_operations.utils.gpu_logger import log_gpu
from model_operations.utils.model_utils import load_model, load_tokenizer

from data_processing.pretokenizers.firstpretokenizer import FirstPretokenizer
from data_processing.utils.data_loader import load_and_split_dataset
from data_processing.utils.data_preparation import preprocess
from datasets import Dataset


def main():
    # Setup
    os.makedirs(GENERATED_OUTPUTS_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    pretokenizer = FirstPretokenizer(_use_dedent=True, _use_semantics=True)

    # Load model and tokenizer
    if FINETUNING:
        model = load_model(FINETUNED_MODEL_DIR, RUN_CUSTOM_LOSS).to(device)
        tokenizer, specifics = load_tokenizer(FINETUNED_MODEL_DIR, pretokenizer)
    else:
        model = load_model(MODEL_NAME, False).to(device)
        tokenizer, _ = load_tokenizer(MODEL_NAME)
        if tokenizer.pad_token is None:
            print("⚠️ GPT-2 tokenizer does not have a pad token. Adding '<pad>' as padding token.")
            tokenizer.add_special_tokens({'pad_token': '<pad>'})
            model.resize_token_embeddings(len(tokenizer))

    # Load test dataset
    raw_test_set = Dataset.load_from_disk(TEST_SPLIT_DIR).select(range(NUM_EXAMPLES_TO_GENERATE))

    # Preprocess test dataset
    tokenized_test_set = raw_test_set.map(
        lambda batch: preprocess(batch, tokenizer, MAX_INPUT_LENGTH, MAX_OUTPUT_LENGTH),
        batched=True,
        remove_columns=raw_test_set.column_names
    )

    # Set logits processor
    logits_processor = None
    if RUN_LOGITS_PROCESSOR and specifics:
        semantic_start_id, semantic_end_id, code_token_ids, semantic_token_ids = specifics

        logits_processor = LogitsProcessorList([
            SemanticCodeLogitsMask(
                semantic_token_ids=semantic_token_ids,
                code_token_ids=code_token_ids,
                semantic_start_id=semantic_start_id,
                semantic_stop_id=semantic_end_id
            )
        ])

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
        logits_processor=logits_processor
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