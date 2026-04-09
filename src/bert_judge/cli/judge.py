import argparse
import json
from pathlib import Path

from ..judges import BERTJudge, LLMJudge, RegexJudge
from ..utils import discover_task_functions, get_model_name, load_json_list, parse_tasks


def require_model_path(args):
	if not args.model_path:
		raise ValueError(f"`--model_path` is required for judge type `{args.judge_type}`.")


def make_judge(args):
	if args.judge_type == "BERTJudge":
		require_model_path(args)
		return BERTJudge(
			model_path=args.model_path,
			trust_remote_code=args.trust_remote_code,
			dtype=args.dtype,
			device_map=args.device_map,
		)

	if args.judge_type == "LLMJudge":
		require_model_path(args)
		return LLMJudge(
			model_path=args.model_path,
			backend=args.backend,
			trust_remote_code=args.trust_remote_code,
			dtype=args.dtype,
			temperature=args.temperature,
			top_p=args.top_p,
			top_k=args.top_k,
			min_p=args.min_p,
			presence_penalty=args.presence_penalty,
			max_tokens=args.max_tokens,
			tensor_parallel_size=args.tensor_parallel_size,
			device_map=args.device_map,
		)

	if args.judge_type == "RegexJudge":
		return RegexJudge(
			pattern=args.pattern,
			metric=args.metric,
		)

	raise ValueError(f"Unsupported judge type: {args.judge_type}")


def load_candidates(candidates_path):
	return load_json_list(candidates_path)


def save_scores(scores, output_dir):
	output_dir.mkdir(parents=True, exist_ok=True)
	scores_path = output_dir / "scores.json"
	scores_path.write_text(json.dumps(scores, indent=2, ensure_ascii=False), encoding="utf-8")
	return scores_path


def build_judge_args_fragment(args):
	if args.judge_type == "BERTJudge":
		return get_model_name(args.model_path)

	if args.judge_type == "LLMJudge":
		judge_args = get_model_name(args.model_path)
		if args.instruction_type != "soft":
			judge_args += f"_{args.instruction_type}"
		return judge_args

	if args.judge_type == "RegexJudge":
		return args.metric

	raise ValueError(f"Unsupported judge type: {args.judge_type}")


def score_task(judge, judge_type, dataset, candidates, args):
	questions = dataset["question"]
	references = dataset["reference"]

	if len(candidates) != len(references):
		raise ValueError(
			f"Length mismatch for task: got {len(candidates)} candidates but {len(references)} references."
		)
	
	if judge_type == "BERTJudge":
		return judge.predict(
			questions=questions,
			candidates=candidates,
			references=references,
			batch_size=args.batch_size,
		)

	if judge_type == "LLMJudge":
		return judge.predict(
			questions=questions,
			candidates=candidates,
			references=references,
			instruction_type=args.instruction_type,
		)
	
	if judge_type == "RegexJudge":
		return judge.predict(
			candidates=candidates,
			references=references,
		)

	raise ValueError(f"Unsupported judge type: {judge_type}")


def build_parser():
	parser = argparse.ArgumentParser(
		description="Score generated candidates with BERTJudge, LLMJudge, or RegexJudge.",
	)

	parser.add_argument(
		"judge_type",
		choices=["BERTJudge", "LLMJudge", "RegexJudge"],
		help="Judge class to use.",
	)
	parser.add_argument(
		"tasks",
		nargs="+",
		help="Task names to score (space-separated and/or comma-separated).",
	)
	parser.add_argument(
		"candidates_dir",
		help="Base directory that contains candidates and where scores will be saved.",
	)

	parser.add_argument(
		"--candidate_model",
		required=True,
		help="Model-name folder under each task (path: candidates_dir/task_name/candidate_model_name).",
	)

	parser.add_argument("--model_path", help="Judge model path or HF model id (required for BERTJudge/LLMJudge).")
	parser.add_argument("--trust_remote_code", action="store_true")
	parser.add_argument("--dtype", default="bfloat16")
	parser.add_argument("--device_map", default="auto")

	parser.add_argument("--batch_size", type=int, default=1, help="Used by BERTJudge.")

	parser.add_argument("--backend", choices=["hf", "vllm"], default="vllm", help="Used by LLMJudge.")
	parser.add_argument("--instruction_type", choices=["strict", "soft"], default="soft", help="Used by LLMJudge.")
	parser.add_argument("--temperature", type=float, default=0.0, help="Used by LLMJudge.")
	parser.add_argument("--top_p", type=float, default=1.0, help="Used by LLMJudge.")
	parser.add_argument("--top_k", type=int, default=-1, help="Used by LLMJudge.")
	parser.add_argument("--min_p", type=float, default=0.0, help="Used by LLMJudge.")
	parser.add_argument("--presence_penalty", type=float, default=0.0, help="Used by LLMJudge.")
	parser.add_argument("--max_tokens", type=int, default=2048, help="Used by LLMJudge.")
	parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Used by LLMJudge with vLLM backend.")

	parser.add_argument("--pattern", default=r"Final answer:\\s*(.+)", help="Used by RegexJudge.")
	parser.add_argument("--metric", choices=["EM", "ROUGE", "Math-Verify"], default="EM", help="Used by RegexJudge.")

	return parser


def main():
	parser = build_parser()
	args = parser.parse_args()
	task_names = parse_tasks(args.tasks)

	if not task_names:
		raise ValueError("No task names provided.")

	task_registry = discover_task_functions()
	unknown_tasks = sorted(set(task_names) - set(task_registry))
	if unknown_tasks:
		available = ", ".join(sorted(task_registry))
		unknown = ", ".join(unknown_tasks)
		raise ValueError(f"Unknown task(s): {unknown}. Available task names: {available}")

	judge = make_judge(args)
	base_dir = Path(args.candidates_dir)
	candidate_model_name = get_model_name(args.candidate_model)

	for task_name in task_names:
		task_dataset = task_registry[task_name]()
		if "question" not in task_dataset.column_names or "reference" not in task_dataset.column_names:
			raise ValueError(
				f"Task `{task_name}` must expose `question` and `reference` columns; got {task_dataset.column_names}."
			)

		task_dir = base_dir / task_name / candidate_model_name
		candidates_path = task_dir / "candidates.json"
		candidates = load_candidates(candidates_path)

		scores = score_task(
			judge=judge,
			judge_type=args.judge_type,
			dataset=task_dataset,
			candidates=candidates,
			args=args,
		)
		judge_args_fragment = build_judge_args_fragment(args)
		scores_output_dir = task_dir / args.judge_type / judge_args_fragment
		scores_path = save_scores(scores, scores_output_dir)
		print(f"Saved {len(scores)} scores to {scores_path}")


if __name__ == "__main__":
	main()
