import argparse
import json
from pathlib import Path

from ..generators import HFGenerator, vLLMGenerator
from ..utils import discover_task_functions, get_model_name, parse_tasks


def make_generator(args):
	common_kwargs = {
		"model_path": args.model_path,
		"trust_remote_code": args.trust_remote_code,
		"dtype": args.dtype,
		"temperature": args.temperature,
		"top_p": args.top_p,
		"top_k": args.top_k,
		"min_p": args.min_p,
		"presence_penalty": args.presence_penalty,
		"max_tokens": args.max_tokens,
	}

	if args.backend == "hf":
		return HFGenerator(
			device_map=args.device_map,
			**common_kwargs,
		)

	if args.backend == "vllm":
		if vLLMGenerator is None:
			raise ImportError(
				"vLLM backend requested but vLLM is not available. Install it with `pip install vllm`."
			)
		return vLLMGenerator(
			tensor_parallel_size=args.tensor_parallel_size,
			**common_kwargs,
		)

	raise ValueError(f"Unsupported backend: {args.backend}")


def save_task_outputs(candidates, output_dir, task_name, model_name):
	destination = Path(output_dir) / task_name / model_name
	destination.mkdir(parents=True, exist_ok=True)
	output_path = destination / "candidates.json"
	output_path.write_text(json.dumps(candidates, indent=2, ensure_ascii=False), encoding="utf-8")
	return output_path


def build_parser():
	parser = argparse.ArgumentParser(
		description="Run generations for one or more tasks and save outputs as JSON.",
	)

	parser.add_argument("model_path", help="Model path or HF model id.")
	parser.add_argument(
		"tasks",
		nargs="+",
		help="Task names to run (space-separated and/or comma-separated), e.g. `gsm8k_test mmlu_test_strict`.",
	)
	parser.add_argument("output_dir", help="Base output directory.")

	parser.add_argument("--backend", choices=["hf", "vllm"], default="vllm")
	parser.add_argument("--trust_remote_code", action="store_true")
	parser.add_argument("--dtype", default="bfloat16")

	parser.add_argument("--temperature", type=float, default=0.0)
	parser.add_argument("--top_p", type=float, default=1.0)
	parser.add_argument("--top_k", type=int, default=-1)
	parser.add_argument("--min_p", type=float, default=0.0)
	parser.add_argument("--presence_penalty", type=float, default=0.0)
	parser.add_argument("--max_tokens", type=int, default=2048)

	parser.add_argument("--batch_size", type=int, default=1, help="Only used by HF backend.")
	parser.add_argument("--device_map", default="auto", help="Only used by HF backend.")
	parser.add_argument(
		"--tensor_parallel_size",
		type=int,
		default=1,
		help="Only used by vLLM backend.",
	)

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

	generator = make_generator(args)
	model_name = get_model_name(args.model_path)

	for task_name in task_names:
		task_fn = task_registry[task_name]
		questions = task_fn()["question"]

		if args.backend == "hf":
			candidates = generator.generate(questions, batch_size=args.batch_size)
		else:
			candidates = generator.generate(questions)

		output_path = save_task_outputs(candidates, args.output_dir, task_name, model_name)
		print(f"Saved {len(candidates)} candidates to {output_path}")


if __name__ == "__main__":
	main()
