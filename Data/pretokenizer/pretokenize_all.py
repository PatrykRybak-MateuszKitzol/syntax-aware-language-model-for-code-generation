import ast
import json
from pretokenizer import pretokenize

preprocessed_data = []

with open('../../../preprocessed_dataset.json', 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():  # skip empty lines
            preprocessed_data.append(json.loads(line))

for example in preprocessed_data:
    code = example["code"]
    try:
        parsed = pretokenize(ast.parse(code), _use_dedent=True, _use_semantics=True)
        example["parsed"] = parsed
    except SyntaxError as e:
        print(f"Syntax error in example {example.get('func_name', '')}: {e}")
        example["parsed"] = None  # or you could skip or log these separately

with open('../../../parsed_dataset.json', 'w') as out_f:
    json.dump(preprocessed_data, out_f, indent=2)