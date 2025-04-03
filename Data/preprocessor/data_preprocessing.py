from datasets import load_dataset
import re
import autopep8
import ast
import subprocess
import tempfile
import hashlib
import os
from functools import partial


def is_valid_python(code):
    """Checks if the Python code is syntactically valid using ast.parse."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False
    except Exception:
        return False


def remove_comments(code):
    """Removes comments from Python code."""
    code = re.sub(r'#[^\n]*', '', code)  # Remove inline comments
    code = re.sub(r'(""".*?""")|(\'\'\'.*?\'\'\')', '', code, flags=re.DOTALL)  # Remove docstrings
    return code


def convert_to_python3(code):
    """Converts Python 2 code to Python 3 using 2to3."""
    refactor = lib2to3.refactor.RefactoringTool(lib2to3.refactor.get_fixers_from_package("lib2to3.fixes"))
    try:
        return str(refactor.refactor_string(code, name="code"))
    except Exception:
        return code  # Return original code if conversion fails


def format_pep8(code):
    """Formats code to follow PEP 8 conventions."""
    return autopep8.fix_code(code)


def is_duplicate(code):
    """Checks if the code snippet is a duplicate using SHA256."""
    code_hash = hashlib.sha256(code.encode()).hexdigest()
    if code_hash in seen_hashes:
        return True
    seen_hashes.add(code_hash)
    return False


def preprocess_function(example):
    example["code"] = remove_comments(example["code"])

    if is_duplicate(example["code"]):
        return None

    """
    if not is_valid_python(example["code"]):
        example["code"] = convert_to_python3(example["code"])
        if not is_valid_python(example["code"]):
            return None
    """

    example["code"] = format_pep8(example["code"])

    if not is_valid_python(example["code"]):
        return None

    return example


kept_indices = []

def preprocess_function_with_index(example, idx):
    global kept_indices

    example["code"] = remove_comments(example["code"])

    if is_duplicate(example["code"]):
        return None

    """
    if not is_valid_python(example["code"]):
        example["code"] = convert_to_python3(example["code"])
        if not is_valid_python(example["code"]):
            return None
    """

    example["code"] = format_pep8(example["code"])

    if not is_valid_python(example["code"]):
        return None

    kept_indices.append(idx)
    return example


if __name__ == "__main__":

    access_token = os.getenv("HUGGINGFACE_TOKEN")

    ds = load_dataset("Nan-Do/code-search-net-python", token=access_token)
    seen_hashes = set()

    two_percent_size = int(1 * len(ds['train']))
    ds['train'] = ds['train'].select(range(two_percent_size))

    valid_rows_before = len(ds["train"])
    ds_cleaned = ds["train"].map(preprocess_function)
    #ds_cleaned = ds["train"].map(partial(preprocess_function_with_index), with_indices=True, num_proc=1)
    valid_rows_after = len(ds_cleaned)

    filtered_percentage = ((valid_rows_before - valid_rows_after) / valid_rows_before) * 100
    print(f"Percentage of filtered out data: {filtered_percentage: .2f}%")

    ds_cleaned.to_json("preprocessed_dataset.json")

    with open("kept_indices.txt", "w") as f:
        for idx in kept_indices:
            f.write(f"{idx}\n")
