"""

"""

import sys
from pathlib import Path

root = Path().resolve().parent.parent
sys.path.insert(0, str(root))

from typing import List
from datasets import load_dataset, Dataset
import re
import ast
import hashlib
import autopep8
from core.preprocessor import BasePreprocessor


class PreprocessingPipeline:
    def __init__(self, preprocessors: List[BasePreprocessor]):
        self.preprocessors = preprocessors

    def apply(self, dataset: Dataset) -> Dataset:
        def process(example):
            for preprocessor in self.preprocessors:
                example = preprocessor.process(example)
                if example is None:
                    return None
            return example

        return dataset.map(process, num_proc=1)


class SyntaxValidator(BasePreprocessor):
    def process(self, example: dict) -> dict | None:
        code = example.get("code", "")
        try:
            ast.parse(code)
            return example
        except Exception:
            return None


class Pep8Formatter(BasePreprocessor):
    def process(self, example: dict) -> dict | None:
        code = example.get("code", "")
        example["code"] = autopep8.fix_code(code)
        return example
    

class DuplicateFilter(BasePreprocessor):
    # required to map dataset wiht single thread
    seen_hashes = set()

    def process(self, example):
        code = example.get("code", "")
        code_hash = hashlib.sha256(code.encode()).hexdigest()

        if code_hash in self.__class__.seen_hashes:
            return None

        self.__class__.seen_hashes.add(code_hash)
        return example


class RemoveComments(BasePreprocessor):
    def process(self, example: dict) -> dict | None:
        code = example.get("code", "")
        code = re.sub(r'#[^\n]*', '', code)  # Remove inline comments
        code = re.sub(r'(""".*?""")|(\'\'\'.*?\'\'\')', '', code, flags=re.DOTALL)  # Remove docstrings
        example["code"] = code
        return example
