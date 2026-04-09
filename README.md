# BERT-as-a-Judge

**BERT-as-a-Judge** is a Python package that uses encoder models as evaluation metrics for generative, reference-based tasks. It supports answer generation, synthetic labeling, model training, and provides tools for straightforward inference, facilitating performance evaluation across a wide range of benchmarks.

---

## Setup 

To install the package from the source, clone the repository and install it in editable mode:

```bash
# Clone the repository
git clone [https://github.com/artefactory/BERT-as-a-Judge.git](https://github.com/artefactory/BERT-as-a-Judge.git)

# Navigate into the project directory
cd BERT-as-a-Judge

# Install the package in editable mode
pip install -e .
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

**Output layout:** `./artifacts/candidates/<task_name>/<model_name>/candidates.json`

### 2. Judging Outputs

Use `judge.py` to evaluate candidate answers.

**Example:**

```zsh
python -m bert_judge.cli.judge \
    --judge_type BERTJudge \
    --model_path hgissbkh/BERTJudge-Free-QCR \
    --tasks arc_easy_test arc_challenge_test mmlu_test \
    --candidates_dir ./artifacts/candidates \
    --candidate_model_name Llama-3.2-1B-Instruct
```

Judging is also possible with:
* **`LLMJudge`** ([source code](src/bert_judge/judges/llm.py))
* **`RegexJudge`** ([source code](src/bert_judge/judges/regex.py))

**Output layout:**
`./artifacts/candidates/<task_name>/<candidate_model_name>/<judge_type>/<judge_args>/scores.json`

### 3. Training BERTJudge

Use `train.py` if you want to train a custom `BERTJudge` model. The common workflow involves three steps:

**Step A: Generate outputs with multiple models on multiple tasks**

```zsh
MODEL_PATHS=(
    "meta-llama/Llama-3.2-1B-Instruct"
    "google/gemma-3-1b-it"
    "Qwen/Qwen3-Embedding-0.6B"
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
    "Qwen3-Embedding-0.6B"
)

for candidate_model in "${CANDIDATE_MODELS[@]}"; do
    python -m bert_judge.cli.judge \ 
        --judge_type LLMJudge \ 
        --model_path nvidia/Llama-3_3-Nemotron-Super-49B-v1_5 \ 
        --tasks arc_easy_train,arc_challenge_train,mmlu_train \ 
        --candidates_dir ./artifacts/candidates \ 
        --candidate_model_name "$candidate_model" \ 
        --backend vllm
done
```

**Step C: Train `BERTJudge` on the generated labels**

```zsh
python -m bert_judge.cli.train \ 
    --model_path EuroBERT/EuroBERT-210m \ 
    --dataset_path ./artifacts/BERTJudge-Dataset \ 
    --tasks arc_easy_train,arc_challenge_train,mmlu_train \ 
    --candidates_dir ./artifacts/candidates \ 
    --candidate_models Llama-3.2-1B-Instruct,gemma-3-1b-it,Qwen3-0.6B \ 
    --label_source LLMJudge/Llama_3_3_Nemotron_Super_49B_v1_5 \ 
    --output_dir ./artifacts/models/BERTJudge-Toy
```

### Inspecting CLI Options

To view all available arguments for the CLI tools, use the `--help` flag:

```zsh
python -m bert_judge.cli.generate --help
python -m bert_judge.cli.judge --help
python -m bert_judge.cli.train --help
```

---

## Python Usage

You can also use the package directly within your Python scripts:

```python
from bert_judge.judges import BERTJudge

# Initialize the judge
judge = BERTJudge(
    model_path="hgissbkh/BERTJudge-Free-QCR",
    trust_remote_code=False,
    dtype="bfloat16",
    device_map="auto",
)

# Predict scores
scores = judge.predict(
    questions=["What is the capital of France?"],
    candidates=["Paris"],
    references=["Paris"],
    batch_size=1,
)

print(scores)
```