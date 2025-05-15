import json
import sys

from pathlib import Path
from config import ORIGINAL_DATASET_PATH, HUMANEVAL_DATASET_PATH
from data_processing.pretokenizers.firstpretokenizer import FirstPretokenizer
import ast

print(ORIGINAL_DATASET_PATH)

root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root))

def strip_imports_and_typing(code: str) -> str:
    tree = ast.parse(code)

    # Remove import statements and type annotations
    new_body = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.FunctionDef):
            node.returns = None
            for arg in node.args.args:
                arg.annotation = None
            for arg in getattr(node.args, 'kwonlyargs', []):
                arg.annotation = None
            if node.args.vararg:
                node.args.vararg.annotation = None
            if node.args.kwarg:
                node.args.kwarg.annotation = None
        new_body.append(node)

    tree.body = new_body
    return ast.unparse(tree)

def extract_docstring(tree: ast.AST) -> str:
    if tree.body and isinstance(tree.body[0], ast.FunctionDef):
        return ast.get_docstring(tree.body[0]) or ""
    return ""

def extract_declaration(parsed):
    if "[BLOCK]" in parsed:
        return parsed.split("[BLOCK]", 1)
    else:
        return "", parsed  # empty declaration, full code as fallback

with open(ORIGINAL_DATASET_PATH, 'r', encoding='utf-8') as infile, open(HUMANEVAL_DATASET_PATH, 'w', encoding='utf-8') as outfile:
    for ix, line in enumerate(infile):
        data = json.loads(line)

        docstring = data.get("docstring", "")
        parsed = data.get("parsed", "")
        function_declaration, parsed_no_declaration = extract_declaration(parsed)

        new_data = {
            "only_docstring": docstring,
            "parsed": parsed_no_declaration,
            "docstring": f"{docstring}\n{function_declaration}"
        }

        if ix < 10:
            pretokenizer = FirstPretokenizer(_use_dedent=True, _use_semantics=True)
            print(new_data["docstring"])
            print("---------------------")
            print(new_data["parsed"])
            #print(pretokenizer.reverse(new_data["docstring_and_function_declaration"]))

        outfile.write(json.dumps(new_data, ensure_ascii=False) + '\n')
