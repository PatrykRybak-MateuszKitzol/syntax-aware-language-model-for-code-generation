import sys
from pathlib import Path

# Add project root to sys.path
root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))

from datasets import load_dataset
from config import DATASET_PATH, TRAIN_SPLIT_PERCENT, SEED

def load_and_split_dataset(file_path=DATASET_PATH, train_split_percent=TRAIN_SPLIT_PERCENT, seed=SEED):
    train_split = f"train[:{train_split_percent}%]"

    dataset = load_dataset("json", data_files=file_path, split=train_split)

    split_dataset = dataset.train_test_split(test_size=0.2, seed=seed)
    test_valid_split = split_dataset["test"].train_test_split(test_size=0.5, seed=seed)

    dataset_dict = {
        "train": split_dataset["train"],
        "validation": test_valid_split["train"],
        "test": test_valid_split["test"]
    }

    return dataset_dict

