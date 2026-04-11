# BERT-as-a-Judge

**BERT-as-a-Judge** is a Python package that uses encoder models as evaluation metrics for generative, reference-based tasks. It supports answer generation, synthetic labeling, model training, and provides tools for straightforward inference, facilitating performance evaluation across a wide range of benchmarks.

---

## Setup 

To install the package from the source, clone the repository and install it in editable mode:

```bash
git clone https://github.com/artefactory/BERT-as-a-Judge.git
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

For smoother training with `transformers.Trainer` (recommended):

```bash
pip install accelerate
```

* For using `RegexJudge`:

```bash
pip install rouge-score math-verify
```

---

## CLI Tools

### 1. Output Generation

Use `generate.py` to run a model on one or more tasks and save the generated candidates.

**Example:**

```zsh
python -m bert_judge.cli.generate \ 
    --model_path meta-llama/Llama-3.2-1B-Instruct \ 
    --tasks arc_easy_test,arc_challenge_test,mmlu_test \ 
    --output_dir ./artifacts/candidates \ 
    --backend vllm
```

### 2. Judging Outputs

Use `judge.py` to evaluate candidate answers.

**Example:**

```zsh
python -m bert_judge.cli.judge \ 
    --judge_type BERTJudge \ 
    --model_path hgissbkh/BERTJudge-Free-QCR \ 
    --trust_remote_code \ 
    --tasks arc_easy_test,arc_challenge_test,mmlu_test \ 
    --candidates_dir ./artifacts/candidates \ 
    --candidate_model Llama-3.2-1B-Instruct
```

Judging is also possible with:
* **`LLMJudge`** ([source code](src/bert_judge/judges/llm.py))
* **`RegexJudge`** ([source code](src/bert_judge/judges/regex.py))

### 3. Training BERTJudge

Use `train.py` if you want to train a custom `BERTJudge` model. The common workflow involves three steps:

**Step A: Generate outputs with multiple models on multiple tasks**

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

**Step B: Generate synthetic labels with a large `LLMJudge`**

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
        --candidate_model "$candidate_model" \ 
        --backend vllm
done
```

**Step C: Train `BERTJudge` on the generated labels**

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

**Optional: Control sampling with `--training_mix`**

You can provide a JSON file to control how many examples are sampled per task and per candidate model during training.

Example file (`./artifacts/training_mix.json`):

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

Then pass it to `train.py`:

```zsh
python -m bert_judge.cli.train \ 
    ...
    --training_mix ./artifacts/training_mix.json 
```

`training_mix` keys should match task names and candidate model split names in your training dataset.

### Inspecting CLI Options

To view all available arguments for the CLI tools, use the `--help` flag:

```zsh
python -m bert_judge.cli.generate --help
python -m bert_judge.cli.judge --help
python -m bert_judge.cli.train --help
```

---

## Adding a New Task

To add a new benchmark/task, create a new Python file in `src/bert_judge/tasks/`.

### Step 1: Create a task module

Example: `src/bert_judge/tasks/my_custom_task.py`

```python
from ..utils import load_dataset


def my_custom_task():
    def process_fn(ex):
        question = ex["question"].strip()
        reference = ex["answer"].strip()
        return {"question": question, "reference": reference}

    return load_dataset(
        path="my_org/my_dataset",   # HF dataset id or local dataset folder name
        name=None,                  # optional config name
        split="test",               # split to use
        process_fn=process_fn,
    )
```

### Step 2: Respect the required output schema

Your task function must return a Hugging Face dataset with at least:

- `question`
- `reference`

Those two fields are required by the generation/judging/training pipelines.

### Step 3: Use the task from CLI

Task functions are auto-discovered from `bert_judge.tasks`, so no registry file needs manual updates.
Once your function exists, call it by its function name:

```zsh
python -m bert_judge.cli.generate \ 
    --model_path meta-llama/Llama-3.2-1B-Instruct \ 
    --tasks my_custom_task \ 
    --output_dir ./artifacts/candidates
```

### Notes

- For training/evaluation splits, you can create separate functions (for example `my_custom_task_train` and `my_custom_task_test`).
- Keep function names unique across files in `src/bert_judge/tasks/`.
- If you use local mirrors, `LOCAL_DATASETS_DIR` is respected by the dataset loading utilities.

---

## Python Usage

You can use the package directly in Python scripts for quick scoring.

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

Example output:

```text
[0.9946, 0.9988, 0.9888, 0.0748, 0.0300, 0.0199]
```