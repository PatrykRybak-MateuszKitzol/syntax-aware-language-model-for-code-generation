import sys
from pathlib import Path

# Add project root to sys.path
root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))

import wandb
import gc
import torch

from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq, set_seed

from config import (
    MODEL_NAME,
    PROJECT_NAME,
    RUN_NAME,
    SEED,
    MAX_INPUT_LENGTH,
    MAX_OUTPUT_LENGTH,
    FINETUNED_MODEL_DIR,
    TRAINING_ARGS
)
from utils.model_utils import load_model, load_tokenizer, save_model
from utils.data_loader import load_and_split_dataset
from utils.data_preparation import preprocess
from utils.metrics import compute_metrics, save_metrics_to_file, average_metrics
from utils.evaluation import evaluate_in_chunks
from utils.gpu_logger import log_gpu

from pretokenizers.firstpretokenizer import FirstPretokenizer
from training.training_additions import T5WithModeLoss, CustomT5Trainer


def main():
    # Setup
    set_seed(SEED)
    wandb.init(project=PROJECT_NAME, name=RUN_NAME)
    pretokenizer = FirstPretokenizer(_use_dedent=True, _use_semantics=True)

    # Load dataset
    dataset_dict = load_and_split_dataset()

    # Load and prepare model with tokenizer
    tokenizer = load_tokenizer(MODEL_NAME)

    semantic_token_ids = [i for i in range(tokenizer.vocab_size) if i not in tokenizer.all_special_ids]
    tags = [v for k, v in pretokenizer.tags.__dict__.items() if not k.startswith("_")]
    tokenizer.add_tokens(tags)
    code_token_ids = [tokenizer.convert_tokens_to_ids(tok) for tok in tags]
    semantic_start_id = tokenizer.convert_tokens_to_ids(pretokenizer.tags.SEMANTIC_START)
    semantic_end_id = tokenizer.convert_tokens_to_ids(pretokenizer.tags.SEMANTIC_END)
    semantic_token_ids.append(semantic_end_id)
    code_token_ids.remove(semantic_end_id)
    code_token_ids.append(tokenizer.convert_tokens_to_ids("</s>"))
    # Not sure about that but just in case
    code_token_ids.append(tokenizer.convert_tokens_to_ids("<pad>")) 

    model = T5WithModeLoss.from_pretrained(
        MODEL_NAME,
    )
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
    trainer = CustomT5Trainer(
        semantic_start_id=semantic_start_id,
        semantic_stop_id=semantic_end_id,
        semantic_token_ids=semantic_token_ids,
        code_token_ids=code_token_ids,
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, tokenizer, pretokenizer),
    )

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
