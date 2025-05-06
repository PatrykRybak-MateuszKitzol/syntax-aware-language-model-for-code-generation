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
    GENERATED_OUTPUTS_DIR
)

from model_operations.generate_evaluate.evaluation import evaluate_in_chunks
from model_operations.generate_evaluate.metrics import compute_metrics, save_metrics_to_file
from model_operations.training.training_additions import SemanticCodeLogitsMask
from model_operations.utils.gpu_logger import log_gpu
from model_operations.utils.model_utils import load_model, load_tokenizer

from data_processing.pretokenizers.firstpretokenizer import FirstPretokenizer
from data_processing.utils.data_loader import load_and_split_dataset
from data_processing.utils.data_preparation import preprocess


def main():
    # Setup
    os.makedirs(, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load fine-tuned model and tokenizer
    model = load_model(FINETUNED_MODEL_DIR, RUN_CUSTOM_LOSS).to(device)
    
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

    # Set logits processor
    logits_processor = None
    if RUN_LOGITS_PROCESSOR:
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