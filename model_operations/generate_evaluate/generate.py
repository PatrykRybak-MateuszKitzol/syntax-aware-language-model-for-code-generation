import json
import sys
import torch
import os
import gc

from pathlib import Path

from human_eval.data import HUMAN_EVAL
from transformers import LogitsProcessorList, NoBadWordsLogitsProcessor

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
    MODEL_NAME,
    USE_CUSTOM_EOS,
    HUMANEVAL
)

from model_operations.generate_evaluate.evaluation import evaluate_in_chunks
from model_operations.generate_evaluate.metrics import compute_metrics, save_metrics_to_file
from model_operations.training.training_additions import SemanticCodeLogitsMask
from model_operations.utils.gpu_logger import log_gpu
from model_operations.utils.model_utils import load_model, load_tokenizer

from data_processing.pretokenizers.firstpretokenizer import FirstPretokenizer
from data_processing.utils.data_preparation import preprocess
from data_processing.utils.humaneval_tests_preprocessing import load_humaneval_dataset
from datasets import Dataset

from human_eval.evaluation import evaluate_functional_correctness

def main():
    # Setup
    os.makedirs(GENERATED_OUTPUTS_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    pretokenizer = FirstPretokenizer(_use_dedent=True, _use_semantics=True)

    # Load model and tokenizer
    if FINETUNING:
        model = load_model(FINETUNED_MODEL_DIR, RUN_CUSTOM_LOSS).to(device)
        tokenizer, specifics = load_tokenizer(FINETUNED_MODEL_DIR, USE_CUSTOM_EOS, pretokenizer)
    else:
        model = load_model(MODEL_NAME, False).to(device)
        tokenizer, _ = load_tokenizer(MODEL_NAME, USE_CUSTOM_EOS)
        if tokenizer.pad_token is None:
            print("⚠️ GPT-2 tokenizer does not have a pad token. Adding '<pad>' as padding token.")
            tokenizer.add_special_tokens({'pad_token': '<pad>'})
            model.resize_token_embeddings(len(tokenizer))

    # Load test dataset
    if HUMANEVAL:
        raw_test_set = load_humaneval_dataset()
    else:
        raw_test_set = Dataset.load_from_disk(TEST_SPLIT_DIR).select(range(NUM_EXAMPLES_TO_GENERATE))

    # Preprocess test dataset
    tokenized_test_set = raw_test_set.map(
        lambda batch: preprocess(batch, tokenizer, MAX_INPUT_LENGTH, MAX_OUTPUT_LENGTH),
        batched=True,
        remove_columns=raw_test_set.column_names
    )

    # Set logits processor
    logits_processor = None
    if RUN_LOGITS_PROCESSOR and specifics and RUN_CUSTOM_LOSS:
        semantic_start_id, semantic_end_id, code_token_ids, semantic_token_ids = specifics

        logits_processor = LogitsProcessorList([
            SemanticCodeLogitsMask(
                semantic_token_ids=semantic_token_ids,
                code_token_ids=code_token_ids,
                semantic_start_id=semantic_start_id,
                semantic_stop_id=semantic_end_id
            ) # below is strictly for T5
        ] + [NoBadWordsLogitsProcessor(bad_words_ids=[[tokenizer.convert_tokens_to_ids('</s>')]])] if USE_CUSTOM_EOS else [])

        print("Logits processor initialized")

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

    predictions = [sample["prediction"] for sample in outputs]
    references = [sample["reference"] for sample in outputs]

    eval_pred = (predictions, references)

    metrics = compute_metrics(eval_pred, tokenizer=None, pretokenizer=None)

    metrics_save_path = os.path.join(GENERATED_OUTPUTS_DIR, f"{'humaneval_' if HUMANEVAL else ''}generated_metrics.json")

    save_metrics_to_file(metrics, metrics_save_path)

    print(f"✅ Saved evaluation metrics to {metrics_save_path}")

    if HUMANEVAL:
        humaneval_generations = {
            str(i): [{
                "task_id": f"HumanEval/{i}",
                "completion": outputs[i]["prediction"],
                "reference": outputs[i]["reference"],
                "docstring": outputs[i]["input"]
            }]
            for i in range(len(outputs))
        }

        output_path = os.path.join(GENERATED_OUTPUTS_DIR, "humaneval_generations.jsonl")
        with open(output_path, "w", encoding="utf-8") as f:
            for task_id, samples in humaneval_generations.items():
                for sample in samples:
                    json_line = json.dumps(sample, ensure_ascii=False)
                    f.write(json_line + "\n")

        pass_at_k = evaluate_functional_correctness(output_path)

        with open(os.path.join(GENERATED_OUTPUTS_DIR, "humaneval_pass@k.json"), "w") as f:
            json.dump(pass_at_k, f, indent=2)


    # Cleanup
    torch.cuda.empty_cache()
    gc.collect()
    log_gpu()


if __name__ == "__main__":
    main()