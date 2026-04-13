# BERT-as-a-Judge

**BERT-as-a-Judge** is the Python package associated with the paper [BERT-as-a-Judge: A Robust Alternative to Lexical Methods for Efficient Reference-Based LLM Evaluation](https://arxiv.org/abs/2604.09497). It uses encoder models as evaluation metrics for generative, reference-based tasks. The package supports answer generation, synthetic labeling, and model training, and it provides tools for straightforward inference, facilitating performance evaluation across a wide range of benchmarks. 

## Table of Contents
- [Setup](#setup)
- [Python Usage](#python-usage)
- [CLI Tools](#cli-tools)
- [Adding a New Task](#adding-a-new-task)
- [Training a BERTJudge Model](#training-a-bertjudge-model)
- [Citation](#citation)

---

## Setup 

To install the package from the source, clone the repository and install it in editable mode:

```bash
git clone [https://github.com/artefactory/BERT-as-a-Judge.git](https://github.com/artefactory/BERT-as-a-Judge.git)
cd BERT-as-a-Judge
pip install -e .
```

Next, install the required dependencies:

```bash
pip install torch transformers datasets tqdm
```

Optional dependencies can be installed depending on your use case:

* For enabling the `vllm` backend (recommended):

```bash
pip install vllm
```

* For smoother training with `transformers.Trainer` (recommended):

```bash
pip install accelerate
```

* For using `RegexJudge`:

```bash
pip install rouge-score math-verify
```

---

## Python Usage

### Basic Usage

You can use the package directly in Python scripts for quick scoring. First, instantiate `BERTJudge`, then define the candidates to score along with the corresponding question(s) and reference(s), and finally compute the scores.

**Example:**

```python
from bert_judge.judges import BERTJudge

# 1) Initialize the judge
judge = BERTJudge(
    model_path="hgissbkh/BERTJudge-Free-QCR",
    trust_remote_code=True,
    dtype="bfloat16",
)

# 2) Define one question, one reference, and several candidate answers
question = "What is the capital of France?"
reference = "Paris"
candidates = [
    "Paris.",
    "The capital of France is Paris.",
    "I'm hesitating between Paris and London. I would say Paris.",
    "London.",
    "The capital of France is London.",
    "I'm hesitating between Paris and London. I would say London.",
]

# 3) Predict scores (one score per candidate)
scores = judge.predict(
    questions=[question] * len(candidates),
    references=[reference] * len(candidates),
    candidates=candidates,
    batch_size=1,
)

print(scores)
```

**Output:**

```text
[0.9946, 0.9988, 0.9888, 0.0748, 0.0300, 0.0199]
```

### Work Locally

If you wish to store models and datasets locally, consider setting `LOCAL_DATASETS_DIR` and `LOCAL_MODELS_DIR` to the corresponding paths. Make sure datasets and models keep the same name as on Hugging Face.

**Example:**

The following code will load `dataset` and `model` from `local/path/to/datasets/my_dataset` and `local/path/to/models/my_model`, respectively.

```python
import os
from bert_judge.utils import load_dataset, load_hf_encoder

os.environ["LOCAL_DATASETS_DIR"] = "local/path/to/datasets"
os.environ["LOCAL_MODELS_DIR"] = "local/path/to/models"

dataset = load_dataset(path="my_org/my_dataset")
model = load_hf_encoder(path="my_org/my_model")
```

---

## CLI Tools

For scalable, end-to-end evaluation, you can use the CLI tools provided in the package.

### Step 1: Output Generation

Evaluating models first requires generating model outputs on predefined tasks, which are implemented in the [`tasks`](src/bert_judge/tasks/) module. Use [`cli.generate`](src/bert_judge/cli/generate.py) to run a model on one or more tasks and save the generated candidates.

**Example:**

```zsh
python -m bert_judge.cli.generate \ 
    --model_path meta-llama/Llama-3.2-1B-Instruct \ 
    --tasks arc_easy_test,arc_challenge_test,mmlu_test \ 
    --output_dir ./artifacts/candidates \ 
    --backend vllm
```

### Step 2: Judging Outputs

Once the candidate answers are generated, they need to be evaluated using a judge module. Use [`cli.judge`](src/bert_judge/cli/judge.py) to run the evaluation.

**Example:**

```zsh
python -m bert_judge.cli.judge \ 
    --judge_type BERTJudge \ 
    --model_path hgissbkh/BERTJudge-Free-QCR \ 
    --trust_remote_code \ 
    --tasks arc_easy_test,arc_challenge_test,mmlu_test \ 
    --candidates_dir ./artifacts/candidates \ 
    --output_dir ./artifacts/scores \ 
    --candidate_model Llama-3.2-1B-Instruct
```

### Notes

- Judging is also possible with:
    - [`LLMJudge`](src/bert_judge/judges/llm.py)
    - [`RegexJudge`](src/bert_judge/judges/regex.py)
- To view all available arguments for the CLI tools, use the `--help` flag.

**Example:**

```zsh
python -m bert_judge.cli.generate --help
```

---

## Adding a New Task

To add a new benchmark or task, create a new Python file in [`tasks`](src/bert_judge/tasks/).

### Step 1: Create a task module

**Example:** `tasks/my_custom_task.py`

```python
from ..utils import load_dataset

def my_custom_task():
    def process_fn(ex):
        question = ex["question"].strip()
        reference = ex["answer"].strip()
        return {"question": question, "reference": reference}

    return load_dataset(
        path="my_org/my_dataset",   # HF dataset id
        name=None,                  # optional config name
        split="test",               # split to use
        process_fn=process_fn,
    )
```

Your task function must return a Hugging Face dataset with at least the following fields:

- `question`
- `reference`

These two fields are required by the generation, judging, and training pipelines.

### Step 2: Use the task from CLI

Task functions are auto-discovered from the [`tasks`](src/bert_judge/tasks/) module, so no registry file needs manual updates. Once your function exists, call it by its function name:

**Example:**

```zsh
python -m bert_judge.cli.generate \ 
    --model_path meta-llama/Llama-3.2-1B-Instruct \ 
    --tasks my_custom_task \ 
    --output_dir ./artifacts/candidates
```

### Notes

- For training and evaluation splits, you can create separate functions (e.g., `my_custom_task_train` and `my_custom_task_test`).
- Keep function names unique across files in [`tasks`](src/bert_judge/tasks/).

---

## Training a BERTJudge Model

If you want to train a custom `BERTJudge` model using your own data, labels, backbone, or training recipe, you can use [`cli.train`](src/bert_judge/cli/train.py). The common workflow involves three steps:

### Step 1: Generate outputs with multiple models on multiple tasks

**Example:**

```zsh
MODEL_PATHS=(
    "meta-llama/Llama-3.2-1B-Instruct"
    "google/gemma-3-1b-it"
    "Qwen/Qwen3-0.6B"
)

for model_path in "${MODEL_PATHS[@]}"; do
    python -m bert_judge.cli.generate \ 
        --model_path "$model_path" \ 
        --tasks arc_easy_train,arc_challenge_train,mmlu_train \ 
        --output_dir ./artifacts/candidates \ 
        --backend vllm
done
```

### Step 2: Generate synthetic labels with a powerful `LLMJudge`

**Example:**

```zsh
CANDIDATE_MODELS=(
    "Llama-3.2-1B-Instruct"
    "gemma-3-1b-it"
    "Qwen3-0.6B"
)

for candidate_model in "${CANDIDATE_MODELS[@]}"; do
    python -m bert_judge.cli.judge \ 
        --judge_type LLMJudge \ 
        --model_path nvidia/Llama-3_3-Nemotron-Super-49B-v1_5 \ 
        --tasks arc_easy_train,arc_challenge_train,mmlu_train \ 
        --candidates_dir ./artifacts/candidates \ 
        --output_dir ./artifacts/scores \ 
        --candidate_model "$candidate_model" \ 
        --backend vllm
done
```

### Step 3: Train `BERTJudge` on the generated labels

**Example:**

```zsh
python -m bert_judge.cli.train \ 
    --model_path EuroBERT/EuroBERT-210m \ 
    --trust_remote_code \ 
    --dataset_path ./artifacts/BERTJudge-Dataset \ 
    --tasks arc_easy_train,arc_challenge_train,mmlu_train \ 
    --candidates_dir ./artifacts/candidates \ 
    --candidate_models Llama-3.2-1B-Instruct,gemma-3-1b-it,Qwen3-0.6B \ 
    --label_source LLMJudge/Llama-3_3-Nemotron-Super-49B-v1_5 \ 
    --output_dir ./artifacts/models/BERTJudge-Toy
```

### Optional: Control sampling with `--training_mix`

You can provide a JSON file to control how many examples are sampled per task and per candidate model during training.

**Example:** `./artifacts/training_mix.json`

```json
{
    "arc_easy_train": {
        "Llama-3.2-1B-Instruct": 500,
        "gemma-3-1b-it": 500
    },
    "mmlu_train": {
        "Llama-3.2-1B-Instruct": 250,
        "Qwen3-0.6B": 250
    }
}
```

Then pass it to [`cli.train`](src/bert_judge/cli/train.py).

```zsh
python -m bert_judge.cli.train \ 
    ...
    --training_mix ./artifacts/training_mix.json 
```

**Note:** `training_mix` keys should precisely match task names and candidate model split names in your training dataset.

---

## Citation

If you find this resource useful for your research, please consider citing our paper:

```bibtex
@misc{gisserotboukhlef2026bertasajudgerobustalternativelexical,
      title={BERT-as-a-Judge: A Robust Alternative to Lexical Methods for Efficient Reference-Based LLM Evaluation}, 
      author={Hippolyte Gisserot-Boukhlef and Nicolas Boizard and Emmanuel Malherbe and Céline Hudelot and Pierre Colombo},
      year={2026},
      eprint={2604.09497},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={[https://arxiv.org/abs/2604.09497](https://arxiv.org/abs/2604.09497)}, 
}
```