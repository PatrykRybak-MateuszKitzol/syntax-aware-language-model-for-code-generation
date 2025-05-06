import sys
from pathlib import Path
from typing import List, Union

# Add project root to sys.path
# root = Path(__file__).resolve().parent.parent
# sys.path.append(str(root))

from datasets import load_dataset
from datasets import Dataset
from config import DATASET_PATH, TRAIN_SPLIT_PERCENT, SEED

def load_and_split_dataset(dataset: Union[str, List] = DATASET_PATH, train_split_percent=TRAIN_SPLIT_PERCENT, seed=SEED):
    train_split = f"train[:{train_split_percent}%]"

    if isinstance(dataset, str):
        dataset = load_dataset("json", data_files=dataset, split=train_split)
    else:
        dataset = Dataset.from_list(dataset, split=train_split)

    split_dataset = dataset.train_test_split(test_size=0.2, seed=seed)
    test_valid_split = split_dataset["test"].train_test_split(test_size=0.5, seed=seed)

    dataset_dict = {
        "train": split_dataset["train"],
        "validation": test_valid_split["train"],
        "test": test_valid_split["test"]
    }

    return dataset_dict

