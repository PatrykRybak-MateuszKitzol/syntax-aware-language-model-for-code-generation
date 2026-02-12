# Syntax-Aware Language Models for Code Generation

**Authors:** Mateusz Kitzol, Patryk Rybak \
**University:** University of Wrocław, Institute of Computer Science \
**Thesis Type:** Engineer's Thesis \
**Supervisor:** dr Paweł Rychlikowski

---

## About the Project

This repository contains the source code and experiments conducted for our Bachelor's thesis on generating Python code using transformer models (specifically based on the T5 architecture).

The main goal of this work was to investigate whether **enriching small language models with syntax-aware techniques (using AST)** allows for higher quality code generation without the need to train massive LLMs from scratch. The project introduces custom preprocessing and tokenization techniques that "teach" the model the underlying structure of the programming language.


## Key Concepts and Solutions

Our approach modifies how the model "sees" and processes source code. Instead of treating code merely as text, we implemented several mechanisms that account for its syntactic structure.

### 1. Data Preprocessing Pipeline
We developed a rigorous data cleaning pipeline based on the **CodeSearchNet (Python)** dataset:
* **Syntax Validation:** Verifies syntactic correctness by parsing code into an AST.
* **PEP-8 Formatting:** Automatically formats code using `autopep8` to ensure stylistic consistency.
* **Deduplication:** Removes duplicates and overly similar functions.
* **Cleaning:** Removes comments and docstrings from function bodies (docstrings serve as the model input).

### 2. Pretokenizer (Syntax-Aware Tokenization)
A core component of our work. ]The Pretokenizer processes Python code by replacing keywords, operators, and structural elements with dedicated **control tokens**, while separating them from user-defined identifiers and literals (**semantic tokens**)

* **Control Tokens:** E.g., `[DEF]`, `[IF]`, `[RETURN]`, `[INDENT]`. These are unique and distinct from natural language vocabulary.
* **Semantic Tokens:** Function names, variables, strings – enclosed within special markers `[SEMANTIC_START]` and `[SEMANTIC_END]`.

**Example of Pretokenizer Output:**

```python
# Original Code:
def add(a, b):
    return a + b

# Pretokenized Representation:
[DEF] [SEMANTIC_START] add [SEMANTIC_END] [DELIMIT_1_L] 
[SEMANTIC_START] a [SEMANTIC_END] [COMMA] 
[SEMANTIC_START] b [SEMANTIC_END] [DELIMIT_1_R] [BLOCK] 
[INDENT] [RETURN] [SEMANTIC_START] a [SEMANTIC_END] 
[ADD] [SEMANTIC_START] b [SEMANTIC_END] [DEDENT]
```


### 3. Segmentator (Syntax-Aware Span Corruption)
Instead of random span masking (as in standard T5 "span corruption"), we created a Segmentator. It leverages the AST to mask **logically coherent code blocks** (e.g., an entire `for` loop, an `if` block, or a single statement). This forces the model to reconstruct the program's structure rather than just random words.

**Example:**
* **Input:** `[DEF] ... <extra_id_0> [BLOCK] [INDENT] [RETURN] ...`
* **Label:** `<extra_id_0> [SEMANTIC_START] func_name [SEMANTIC_END] ...`

### 4. Custom Loss & Logits Processor (Experimental)
This approach was designed to enforce a strict separation within the model's embedding space, effectively creating an **illusion of two independent subspaces**: one for **Control Tokens** (syntax) and one for **Semantic Tokens** (identifiers/natural language).

The primary goal was to prevent the learning of new syntactic tokens from interfering with the pre-trained semantic relationships inherent in the T5 model.

* **Custom Loss (Training):** A selective loss function designed to isolate the learning process. It ensures that the vector positioning of **both** Control and Semantic tokens evolves independently within their respective subspaces, minimizing mutual interference and preventing the degradation of the model's pre-existing natural language knowledge.
* **Logits Processor (Inference):** Maintains the illusion of disjoint spaces during generation. It dynamically masks tokens from the "inactive" subspace, forcing the model to strictly adhere to the current generation mode (Control vs. Semantic).
* **Mode Switching:** Two special tokens (`[SEMANTIC_START]` and `[SEMANTIC_STOP]`) were introduced to allow the model to autonomously decide when to transition between these two embedding subspaces.

*(Note: While theoretically sound, this module did not yield the expected performance gains in our experiments, likely due to challenges in learning the mode-switching behavior, but it remains a significant research component of the thesis).*

## Experiments & Results

Using **T5-Large** as our base, we conducted several comparative experiments.

### Configurations:
1.  **T5 Baseline:** Vanilla T5-Large (without code fine-tuning).
2.  **Finetuned T5:** Standard fine-tuning on (docstring -> code) pairs.
3.  **Pretokenizer T5:** Fine-tuning using our syntax-aware token representation.
4.  **Segmentator T5:** Training with structural masking.

### Results (Metrics):

The **Pretokenizer T5** variant achieved the best results. Introducing syntax awareness significantly improved the quality of generated code compared to the baseline.

| Model | BLEU | ROUGE-L | Precision@1 |
| :--- | :---: | :---: | :---: |
| T5 Baseline | 0.017 | 0.140 | 0.150 |
| Finetuned T5 | 0.178 | 0.357 | 0.499 |
| **Pretokenizer T5 (Best)** | **0.210** | **0.377** | **0.532** |
| Segmentator T5 | 0.202 | 0.368 | 0.517 |

*[Data source: Table 4.1 and Table 4.2 in the thesis]*


### HumanEval Benchmark
Despite the modest size of the model (T5-Large), our best model was able to solve **one task** from the challenging HumanEval benchmark, indicating emerging semantic competence (understanding code logic rather than just mimicking syntax).

**Example of a Correct Solution Generated by the Model:**
*Task:* Concatenate a list of strings.
*Expected solution:*
```python
" ".join(strings)
```
*Model Solution:*
```python
def concatenate ( strings ) :
    return " " . join ([ s for s in strings if s . endswith ( " " ) ])
```

## Repository Structure

The code is organized into modules to facilitate experimentation:

* `core/` – Abstract classes and interfaces for tokenizers and segmentators.
* `data_processing/` – Implementation of the data processing pipeline:
    * `pretokenizer.py` – AST visitor implementation.
    * `segmentator.py` – Logic for extracting code spans.
    * `pipeline.py` – Cleaning and formatting scripts (PEP8).
* `model_operations/` – Training and evaluation scripts, including Custom Loss and Logits Processor implementations.
* `config.py` – Configuration for training parameters and preprocessing.

---
*This repository is part of a Engineer's thesis defended in June 2025.*