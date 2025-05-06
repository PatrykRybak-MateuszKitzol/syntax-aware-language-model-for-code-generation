import ast
import json
import sys

from pathlib import Path
from datasets import Dataset
from typing import Union

# root = Path().resolve().parent
# sys.path.insert(0, str(root))

from data_processing.pretokenizers.firstpretokenizer import FirstPretokenizer


def pretokenize_all(preprocessed_data: Union[str, Dataset], pretokenizer: FirstPretokenizer, save_path: str = None):
    pretokenized_data = []

    if isinstance(preprocessed_data, str):
        with open(preprocessed_data, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # skip empty lines
                    pretokenized_data.append(json.loads(line))
    else:
        for sample in preprocessed_data:
            pretokenized_data.append(sample)

    for example in pretokenized_data:
        code = example["code"]
        try:
            parsed = pretokenizer.pretokenize(ast.parse(code))
            example["parsed"] = parsed
        except SyntaxError as e:
            print(f"Syntax error in example {example.get('func_name', '')}: {e}")
            example["parsed"] = None  # or you could skip or log these separately
    
    if save_path:
        with open(save_path, 'w') as out_f:
            json.dump(pretokenized_data, out_f, indent=2)

    return pretokenized_data
