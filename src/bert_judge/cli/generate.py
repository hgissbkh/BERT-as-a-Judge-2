import argparse
import json
import logging
from pathlib import Path
from typing import Any

from ..generators import HFGenerator, vLLMGenerator
from ..utils import (
	build_output_model_name,
	discover_task_functions,
	get_model_name,
	parse_tasks,
)

LOGGER = logging.getLogger(__name__)


def make_generator(args: argparse.Namespace) -> Any:
	"""Instantiate generation backend from CLI arguments."""
	common_kwargs = {
		"model_path": args.model_path,
		"trust_remote_code": args.trust_remote_code,
		"dtype": args.dtype,
		"enable_thinking": args.enable_thinking,
		"think_token": args.think_token,
		"temperature": args.temperature,
		"top_p": args.top_p,
		"top_k": args.top_k,
		"min_p": args.min_p,
		"presence_penalty": args.presence_penalty,
		"max_tokens": args.max_tokens,
	}

	if args.backend == "hf":
		return HFGenerator(**common_kwargs)

	if args.backend == "vllm":
		if vLLMGenerator is None:
			raise ImportError(
				"vLLM backend requested but vLLM is not available. Install it with `pip install vllm`."
			)
		return vLLMGenerator(
			tensor_parallel_size=args.tensor_parallel_size,
			**common_kwargs,
		)

	raise ValueError(f"Unsupported backend: {args.backend}. Expected one of ['hf', 'vllm'].")


def save_task_outputs(
	candidates: list[str], output_dir: str, task_name: str, model_name: str
) -> Path:
	"""Persist candidates for a task/model pair."""
	destination = Path(output_dir) / task_name / model_name
	destination.mkdir(parents=True, exist_ok=True)
	output_path = destination / "candidates.json"
	output_path.write_text(json.dumps(candidates, indent=2, ensure_ascii=False), encoding="utf-8")
	return output_path


def build_parser() -> argparse.ArgumentParser:
	"""Create argument parser for the generation CLI."""
	parser = argparse.ArgumentParser(
		description="Run generations for one or more tasks and save outputs as JSON.",
	)

	parser.add_argument("--model_path", required=True, help="Model path or HF model id.")
	parser.add_argument(
		"--tasks",
		nargs="+",
		required=True,
		help="Task names to run (space-separated and/or comma-separated), e.g. `gsm8k_test mmlu_test_strict`.",
	)
	parser.add_argument("--output_dir", required=True, help="Base output directory.")
	parser.add_argument("--backend", choices=["hf", "vllm"], default="vllm")
	parser.add_argument("--trust_remote_code", action="store_true")
	parser.add_argument("--dtype", default="bfloat16")
	parser.add_argument("--enable_thinking", action="store_true")
	parser.add_argument(
		"--think_token", type=str, default="</think>", help="Closing thinking token."
	)
	parser.add_argument("--temperature", type=float, default=0.0)
	parser.add_argument("--top_p", type=float, default=1.0)
	parser.add_argument("--top_k", type=int, default=-1)
	parser.add_argument("--min_p", type=float, default=0.0)
	parser.add_argument("--presence_penalty", type=float, default=0.0)
	parser.add_argument("--max_tokens", type=int, default=2048)
	parser.add_argument("--batch_size", type=int, default=1, help="Only used by HF backend.")
	parser.add_argument(
		"--tensor_parallel_size",
		type=int,
		default=1,
		help="Only used by vLLM backend.",
	)

	return parser


def main() -> None:
	"""Run generation workflow across requested tasks."""
	logging.basicConfig(
		level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
	)
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
	output_model_name = build_output_model_name(
		model_name=get_model_name(args.model_path),
		temperature=args.temperature,
		top_p=args.top_p,
		top_k=args.top_k,
		min_p=args.min_p,
		presence_penalty=args.presence_penalty,
		max_tokens=args.max_tokens,
		enable_thinking=args.enable_thinking,
	)

	for task_name in task_names:
		task_fn = task_registry[task_name]
		questions = task_fn()["question"]
		LOGGER.info("Generating outputs for task '%s' (%d prompts)", task_name, len(questions))

		if args.backend == "hf":
			candidates = generator.generate(questions, batch_size=args.batch_size)
		else:
			candidates = generator.generate(questions)

		output_path = save_task_outputs(candidates, args.output_dir, task_name, output_model_name)
		LOGGER.info("Saved %d candidates to %s", len(candidates), output_path)


if __name__ == "__main__":
	main()
