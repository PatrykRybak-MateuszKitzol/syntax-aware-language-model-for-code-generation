import os

# === Basic Training Setup ===
MODEL_NAME = "t5-base"
TRAIN_SPLIT_PERCENT = 1
NUM_EPOCHS = 1
FINETUNING = True

# === Input/Output lengths ===
MAX_INPUT_LENGTH = 256
MAX_OUTPUT_LENGTH = 512  # Maximum output length that t5-base supports (93% of your data fits without segmentation)

# === Training method settings ===
RUN_SEGEMENTATOR = False
RUN_CUSTOM_LOSS = False
RUN_LOGITS_PROCESSOR = False # Whether to use the logits processor (SemanticCodeLogitsMask)
USE_CUSTOM_EOS = True
EOS = "<custom_eos>"

# === Generation Hyperparameters ===
GENERATION_ARGS = {
    "max_length": MAX_OUTPUT_LENGTH,
    #"num_beams": 1,
    #"no_repeat_ngram_size": 0,
    #"early_stopping": True,
    #"length_penalty": 1,
}



# === Output directories ===
# Get the root of the project
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))

METHOD_FLAGS = f"loss{RUN_CUSTOM_LOSS}"

# Base experiment name with method flags included
if FINETUNING:
    BASE_NAME = f"{MODEL_NAME}-split{TRAIN_SPLIT_PERCENT}-epochs{NUM_EPOCHS}-{METHOD_FLAGS}"
else:
    BASE_NAME = MODEL_NAME

# Final experiment directory (EXCLUDING generation args)
EXPERIMENT_DIR = os.path.join(PROJECT_ROOT, "model_operations", "training", "models", BASE_NAME)
CHECKPOINTS_DIR = os.path.join(EXPERIMENT_DIR, "checkpoints")

# Generation args string (used ONLY in generated outputs dir)
GEN_ARGS_STRING = "-".join(f"{k}{v}" for k, v in GENERATION_ARGS.items())
GENERATED_OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "model_operations", "generate_evaluate", "generations", BASE_NAME, f"{GEN_ARGS_STRING}_logits{RUN_LOGITS_PROCESSOR}")

SAVE_OUTPUTS_PATH = os.path.join(GENERATED_OUTPUTS_DIR, "generated_outputs.json")
FINETUNED_MODEL_DIR = CHECKPOINTS_DIR

# New folder to save the dataset splits
SPLITTED_DATASET_DIR = os.path.join(EXPERIMENT_DIR, "splitted_dataset")
TRAIN_SPLIT_DIR = os.path.join(SPLITTED_DATASET_DIR, "train_split")
VALIDATION_SPLIT_DIR = os.path.join(SPLITTED_DATASET_DIR, "validation_split")
TEST_SPLIT_DIR = os.path.join(SPLITTED_DATASET_DIR, "test_split")


print(EXPERIMENT_DIR)



# === W&B (Weights & Biases) project tracking ===
PROJECT_NAME = "syntax-aware-language-model-for-code-generation"
RUN_NAME = f"{MODEL_NAME}-split{TRAIN_SPLIT_PERCENT}-epochs{NUM_EPOCHS}-doc2code-run"

# === Dataset path ===
DATASET_PATH = "docstring_and_code.jsonl"

# === Default random seed ===
SEED = 42

# === Training Hyperparameters ===
TRAINING_ARGS = {
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "gradient_accumulation_steps": 2,
    "output_dir": CHECKPOINTS_DIR,
    "num_train_epochs": NUM_EPOCHS,
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
    "eval_steps": 500,
    "save_steps": 500,
    "save_total_limit": 2,
    "logging_steps": 100,
    "bf16": True,
    "report_to": "wandb",
    "run_name": RUN_NAME,
}

# === Generation Settings ===
NUM_EXAMPLES_TO_GENERATE = 5  # Number of examples to use from test set
CHUNK_SIZE = 5  # How many examples to generate at once
