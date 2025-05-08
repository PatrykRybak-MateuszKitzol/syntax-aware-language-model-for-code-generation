import sys
import wandb
import gc
import torch
import os

from pathlib import Path
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq, set_seed

root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root))

from config import (
    RUN_CUSTOM_LOSS,
    MODEL_NAME,
    PROJECT_NAME,
    RUN_NAME,
    SEED,
    MAX_INPUT_LENGTH,
    MAX_OUTPUT_LENGTH,
    FINETUNED_MODEL_DIR,
    TRAINING_ARGS,
    TRAIN_SPLIT_DIR,
    VALIDATION_SPLIT_DIR,
    TEST_SPLIT_DIR
)

from model_operations.training.training_additions import T5WithModeLoss, CustomT5Trainer
from model_operations.generate_evaluate.metrics import compute_metrics, save_metrics_to_file, average_metrics
from model_operations.generate_evaluate.evaluation import evaluate_in_chunks
from model_operations.utils.gpu_logger import log_gpu
from model_operations.utils.model_utils import load_model, load_tokenizer, save_model

from data_processing.pretokenizers.firstpretokenizer import FirstPretokenizer
from data_processing.utils.data_loader import load_and_split_dataset
from data_processing.utils.data_preparation import preprocess


def main():
    # Setup
    set_seed(SEED)
    wandb.init(project=PROJECT_NAME, name=RUN_NAME)
    pretokenizer = FirstPretokenizer(_use_dedent=True, _use_semantics=True)

    # Load dataset
    dataset_dict = load_and_split_dataset()

    # Ensure the directories exist
    os.makedirs(TRAIN_SPLIT_DIR, exist_ok=True)
    os.makedirs(VALIDATION_SPLIT_DIR, exist_ok=True)
    os.makedirs(TEST_SPLIT_DIR, exist_ok=True)

    # Save the datasets to disk
    dataset_dict["train"].save_to_disk(TRAIN_SPLIT_DIR)
    dataset_dict["validation"].save_to_disk(VALIDATION_SPLIT_DIR)
    dataset_dict["test"].save_to_disk(TEST_SPLIT_DIR)

    # Load and prepare model with tokenizer
    tokenizer, specifics = load_tokenizer(MODEL_NAME, pretokenizer)
    if specifics:
        semantic_start_id, semantic_end_id, code_token_ids, semantic_token_ids = specifics

    model = load_model(MODEL_NAME, RUN_CUSTOM_LOSS)
    model.resize_token_embeddings(len(tokenizer))

    # Preprocess dataset
    tokenized_dataset = {
        split: dataset.map(
            lambda batch: preprocess(batch, tokenizer, MAX_INPUT_LENGTH, MAX_OUTPUT_LENGTH),
            batched=True,
            remove_columns=dataset.column_names
        )
        for split, dataset in dataset_dict.items()
    }

    # Data collator (dynamic padding)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Training configuration
    training_args = TrainingArguments(**TRAINING_ARGS)

    # Trainer setup
    trainer_args = {
        'model': model,
        'args': training_args,
        'train_dataset': tokenized_dataset["train"],
        'eval_dataset': tokenized_dataset["validation"],
        'tokenizer': tokenizer,
        'data_collator': data_collator,
        'compute_metrics': lambda p: compute_metrics(p, tokenizer, pretokenizer),
    }
    if RUN_CUSTOM_LOSS and specifics:
        trainer_args['semantic_start_id'] = semantic_start_id
        trainer_args['semantic_stop_id'] = semantic_end_id
        trainer_args['semantic_token_ids'] = semantic_token_ids
        trainer_args['code_token_ids'] = code_token_ids

        trainer = CustomT5Trainer(**trainer_args)
    else:
        trainer = Trainer(**trainer_args)

    # === Train ===
    trainer.train()

    # Save fine-tuned model
    save_model(model, tokenizer, output_dir=FINETUNED_MODEL_DIR)

    # Clean up
    wandb.finish()
    torch.cuda.empty_cache()
    gc.collect()
    log_gpu()



if __name__ == "__main__":
    main()
