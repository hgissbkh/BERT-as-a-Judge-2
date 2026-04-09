import argparse
import json
from pathlib import Path

from datasets import Dataset, DatasetDict

from ..judges import BERTJudge
from ..utils import discover_task_functions, get_model_name, load_dataset, load_json_list, parse_tasks


def normalize_report_to(report_to_values):
	if report_to_values is None:
		return None
	items = []
	for value in report_to_values:
		for item in value.split(","):
			item = item.strip()
			if item:
				items.append(item)
	return items


def parse_training_mix(training_mix):
	if training_mix is None:
		return None
	path = Path(training_mix)
	if not path.exists():
		raise FileNotFoundError(f"training_mix file not found: {path}")
	return json.loads(path.read_text(encoding="utf-8"))


def build_task_training_dataset(task_name, task_fn, candidates_dir, candidate_model_name, label_source):
	task_dataset = task_fn()
	if "question" not in task_dataset.column_names or "reference" not in task_dataset.column_names:
		raise ValueError(
			f"Task `{task_name}` must expose `question` and `reference`; got {task_dataset.column_names}."
		)

	sanitized_model_name = get_model_name(candidate_model_name)
	task_dir = Path(candidates_dir) / task_name / sanitized_model_name
	candidates = load_json_list(task_dir / "candidates.json")
	scores = load_json_list(task_dir / label_source.strip("/") / "scores.json")

	questions = task_dataset["question"]
	references = task_dataset["reference"]

	lengths = {
		"question": len(questions),
		"reference": len(references),
		"candidate": len(candidates),
		"label": len(scores),
	}
	if len(set(lengths.values())) != 1:
		raise ValueError(f"Length mismatch for task `{task_name}`: {lengths}")

	return Dataset.from_dict(
		{
			"question": questions,
			"reference": references,
			"candidate": candidates,
			"label": scores,
		}
	)


def build_training_dataset(task_names, task_registry, candidates_dir, candidate_models, label_source):
	datasets = {}
	for task_name in task_names:
		task_fn = task_registry[task_name]
		splits = {}
		for candidate_model_name in candidate_models:
			sanitized_model_name = get_model_name(candidate_model_name)
			task_dataset = build_task_training_dataset(
				task_name=task_name,
				task_fn=task_fn,
				candidates_dir=candidates_dir,
				candidate_model_name=candidate_model_name,
				label_source=label_source,
			)
			splits[sanitized_model_name] = task_dataset
		datasets[task_name] = splits

	return datasets


def load_training_dataset(dataset_path):
	dataset_path = Path(dataset_path)
	if not dataset_path.exists():
		raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

	datasets = {}
	for task_dir in sorted(dataset_path.iterdir()):
		if not task_dir.is_dir():
			continue

		splits = {}
		for split_dir in sorted(task_dir.iterdir()):
			if not split_dir.is_dir():
				continue
			split_name = split_dir.name
			splits[split_name] = load_dataset(str(task_dir), split=split_name)

		if splits:
			datasets[task_dir.name] = DatasetDict(splits)

	if not datasets:
		raise ValueError(f"No task datasets found under: {dataset_path}")

	return datasets


def save_training_dataset(datasets, dataset_path):
	dataset_path = Path(dataset_path)
	dataset_path.mkdir(parents=True, exist_ok=True)
	for task_name, task_dataset_dict in datasets.items():
		task_output_path = dataset_path / task_name
		task_dataset_dict.save_to_disk(str(task_output_path))


def build_parser():
	parser = argparse.ArgumentParser(
		description="Train a BERTJudge from task datasets + candidates + selected judge scores.",
	)

	parser.add_argument("model_path", help="BERT judge model path or HF model id.")
	parser.add_argument(
		"tasks",
		nargs="+",
		help="Task names to use for training (space-separated or comma-separated).",
	)
	parser.add_argument("candidates_dir", help="Base directory containing candidates and judge scores.")
	parser.add_argument("output_dir", help="Output directory for trained BERTJudge model.")
	parser.add_argument(
		"--dataset_path",
		required=True,
		help="Path for cached training datasets. If it exists, data is loaded from it; otherwise built and saved there.",
	)

	parser.add_argument(
		"--candidate_models",
		nargs="+",
		default=None,
		help="Candidate model names to use for training (space or comma-separated)",
	)
	parser.add_argument(
		"--label_source",
		default=None,
		help="Judge score selector path under each task/model, e.g. `LLMJudge/'Llama_3_3_Nemotron_Super_49B_v1_5'_soft`.",
	)

	parser.add_argument("--trust_remote_code", action="store_true")
	parser.add_argument("--dtype", default="bfloat16")
	parser.add_argument("--device_map", default="auto")

	parser.add_argument("--include_question", action=argparse.BooleanOptionalAction, default=True)
	parser.add_argument("--training_mix", default=None, help="Path to a JSON file for training mix (same structure as BERTJudge.fit).")
	parser.add_argument("--num_train_epochs", type=float, default=1)
	parser.add_argument("--batch_size", type=int, default=4)
	parser.add_argument("--learning_rate", type=float, default=2e-5)
	parser.add_argument("--warmup_ratio", type=float, default=0.05)
	parser.add_argument("--lr_scheduler_type", default="linear")
	parser.add_argument("--logging_strategy", default="steps")
	parser.add_argument("--logging_steps", type=int, default=10)
	parser.add_argument("--logging_dir", default=None)
	parser.add_argument(
		"--report_to",
		nargs="+",
		default=None,
		help="List or comma-separated list of reporting integrations (same as BERTJudge.fit).",
	)
	parser.add_argument("--save_strategy", default="steps")
	parser.add_argument("--save_steps", type=int, default=500)
	parser.add_argument("--save_total_limit", type=int, default=1)
	parser.add_argument("--seed", type=int, default=0)

	return parser


def main():
	parser = build_parser()
	args = parser.parse_args()

	dataset_path = Path(args.dataset_path)
	if dataset_path.exists():
		train_dataset = load_training_dataset(dataset_path)
	else:
		task_names = parse_tasks(args.tasks)
		if not task_names:
			raise ValueError("No task names provided.")

		task_registry = discover_task_functions()
		unknown_tasks = sorted(set(task_names) - set(task_registry))
		if unknown_tasks:
			available = ", ".join(sorted(task_registry))
			unknown = ", ".join(unknown_tasks)
			raise ValueError(f"Unknown task(s): {unknown}. Available task names: {available}")

		if args.label_source is None:
			raise ValueError("`--label_source` is required when building dataset from candidates/scores.")

		candidate_models = parse_tasks(args.candidate_models or [])
		if not candidate_models:
			raise ValueError("`--candidate_models` is required when building dataset from candidates/scores.")

		train_dataset = build_training_dataset(
			task_names=task_names,
			task_registry=task_registry,
			candidates_dir=args.candidates_dir,
			candidate_models=candidate_models,
			label_source=args.label_source,
		)
		save_training_dataset(train_dataset, dataset_path)

	judge = BERTJudge(
		model_path=args.model_path,
		trust_remote_code=args.trust_remote_code,
		dtype=args.dtype,
		device_map=args.device_map,
	)

	training_mix = parse_training_mix(args.training_mix)
	report_to = normalize_report_to(args.report_to)

	judge.fit(
		dataset=train_dataset,
		output_dir=args.output_dir,
		include_question=args.include_question,
		training_mix=training_mix,
		num_train_epochs=args.num_train_epochs,
		batch_size=args.batch_size,
		learning_rate=args.learning_rate,
		warmup_ratio=args.warmup_ratio,
		lr_scheduler_type=args.lr_scheduler_type,
		logging_strategy=args.logging_strategy,
		logging_steps=args.logging_steps,
		logging_dir=args.logging_dir,
		report_to=report_to,
		save_strategy=args.save_strategy,
		save_steps=args.save_steps,
		save_total_limit=args.save_total_limit,
		seed=args.seed,
	)

	print(f"Training completed. Model saved to {args.output_dir}")


if __name__ == "__main__":
	main()
