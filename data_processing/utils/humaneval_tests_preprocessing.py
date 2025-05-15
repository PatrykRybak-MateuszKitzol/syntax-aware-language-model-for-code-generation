import ast
from datasets import Dataset
from human_eval.data import read_problems
from data_processing.pretokenizers.firstpretokenizer import FirstPretokenizer

pretokenizer = FirstPretokenizer(_use_dedent=True, _use_semantics=True)


def strip_imports_and_typing(code: str) -> str:
    tree = ast.parse(code)

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
        docstring = ast.get_docstring(tree.body[0]) or ""
        lines = docstring.splitlines()

        # Keep only lines before the first line starting with >>>
        cleaned_lines = []
        for line in lines:
            if line.strip().startswith(">>>"):
                break
            cleaned_lines.append(line)

        return "\n".join(cleaned_lines).strip()
    return ""



def extract_declaration(parsed: str) -> str:
    return parsed.split("[BLOCK]", 1) if "[BLOCK]" in parsed else parsed


def load_humaneval_dataset():
    problems = read_problems()
    docstrings = []
    parsed_codes = []
    declarations = []

    for k, prob in problems.items():
        prompt = prob["prompt"]
        reference = prob["canonical_solution"]

        try:
            cleaned_prompt = strip_imports_and_typing(prompt)
            tree = ast.parse(cleaned_prompt)
            docstring = extract_docstring(tree).strip()

            pretokenized_prompt = pretokenizer.pretokenize(tree)
            function_declaration = extract_declaration(pretokenized_prompt)

            full_combined = f"{docstring}\n{function_declaration}"

            docstrings.append(prompt.strip())
            parsed_codes.append(reference)
            declarations.append(full_combined)

        except Exception as e:
            print(f"[!] Error processing {k}: {e}")
            continue

    print(declarations[0])

    return Dataset.from_dict({
        "only_docstring": docstrings,
        "parsed": parsed_codes,
        "docstring": declarations
    })


load_humaneval_dataset()