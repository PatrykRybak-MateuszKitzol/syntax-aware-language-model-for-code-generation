import ast
import json

import sys
from pathlib import Path

root = Path().resolve().parent
sys.path.insert(0, str(root))

from pretokenizers.firstpretokenizer import FirstPretokenizer


preprocessed_data = []
pretokenizer = FirstPretokenizer(_use_dedent=False, _use_semantics=True)

with open('../../../preprocessed_dataset.json', 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():  # skip empty lines
            preprocessed_data.append(json.loads(line))

for example in preprocessed_data:
    code = example["code"]
    try:
        parsed = pretokenizer.pretokenize(ast.parse(code), _use_dedent=True, _use_semantics=True)
        example["parsed"] = parsed
    except SyntaxError as e:
        print(f"Syntax error in example {example.get('func_name', '')}: {e}")
        example["parsed"] = None  # or you could skip or log these separately

with open('../../../parsed_dataset.json', 'w') as out_f:
    json.dump(preprocessed_data, out_f, indent=2)