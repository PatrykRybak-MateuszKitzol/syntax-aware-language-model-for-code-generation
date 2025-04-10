from datasets import load_dataset
import re
import autopep8
import ast
import hashlib
import os


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

    example["code"] = format_pep8(example["code"])

    if not is_valid_python(example["code"]):
        return None

    return example


kept_indices = []

def preprocess_function_with_index(example, indices):
    global kept_indices

    example["code"] = remove_comments(example["code"])

    if is_duplicate(example["code"]):
        return None

    example["code"] = format_pep8(example["code"])

    if not is_valid_python(example["code"]):
        return None

    kept_indices.append(indices)
    return example


if __name__ == "__main__":

    access_token = os.getenv("HUGGINGFACE_TOKEN")

    ds = load_dataset("Nan-Do/code-search-net-python", token=access_token)
    seen_hashes = set()

    two_percent_size = int(1 * len(ds['train']))
    ds['train'] = ds['train'].select(range(two_percent_size))

    valid_rows_before = len(ds["train"])
    ds_cleaned = ds["train"].map(preprocess_function)
    valid_rows_after = len(ds_cleaned)

    filtered_percentage = ((valid_rows_before - valid_rows_after) / valid_rows_before) * 100
    print(f"Percentage of filtered out data: {filtered_percentage: .2f}%")

    ds_cleaned.to_json("preprocessed_dataset.json")

    with open("kept_indices.txt", "w") as f:
        for idx in kept_indices:
            f.write(f"{idx}\n")
