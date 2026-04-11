import importlib
import json
import logging
import os
import pkgutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from datasets import (
	Dataset,
	DatasetDict,
	concatenate_datasets,
	load_from_disk,
)
from datasets import (
	get_dataset_config_names as _get_dataset_config_names,
)
from datasets import (
	load_dataset as _load_dataset,
)
from transformers import (
	AutoModelForCausalLM,
	AutoModelForSequenceClassification,
	AutoTokenizer,
)

LOGGER = logging.getLogger(__name__)


def get_dataset_config_names(path: str) -> list[str]:
	"""Return available configuration names for a dataset path."""
	path = resolve_dataset_path(path)
	return _get_dataset_config_names(path)


def load_dataset_dict(
	path: str,
	name: str | list[str] | None = None,
	split: str | list[str] | None = None,
) -> DatasetDict:
	"""Load one or multiple datasets as a dict of `DatasetDict`.

	Args:
		path: Dataset identifier or local path.
		name: Optional config name(s).
		split: Optional split name(s).

	Returns:
		A dictionary mapping each `name` to a `DatasetDict` containing
		the requested splits.
	"""
	path = resolve_dataset_path(path)
	names = name if isinstance(name, list) else [name]
	splits = split if isinstance(split, list) else [split]
	dataset = {}

	try:
		for name in names:
			dataset_dict = DatasetDict()
			for split in splits:
				lfd_path = (
					path
					+ f"/{name}" * bool(name)
					+ f"/{split}" * bool(split)
				)
				if split is None:
					dataset_dict = load_from_disk(lfd_path)
				else:
					dataset_dict[split] = load_from_disk(lfd_path)
			dataset[name] = dataset_dict

	except Exception:
		for name in names:
			dataset_dict = DatasetDict()
			for split in splits:
				ld_kwargs = {
					"path": path,
					"split": split,
					**({"name": name} if name is not None else {}),
				}
				if split is None:
					dataset_dict = _load_dataset(**ld_kwargs)
				else:
					dataset_dict[split] = _load_dataset(**ld_kwargs)
			dataset[name] = dataset_dict

	return dataset


def load_dataset(
	path: str,
	name: str | list[str] | None = None,
	split: str | list[str] | None = None,
	filter_fn: Callable[[dict[str, Any]], bool] | None = None,
	process_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> Dataset:
	"""Load and concatenate one or multiple datasets from disk or HF hub.

	Args:
		path: Dataset identifier or local path.
		name: Optional config name(s).
		split: Optional split name(s).
		filter_fn: Optional filtering callable applied after loading.
		process_fn: Optional mapping callable applied after filtering.

	Returns:
		A concatenated Hugging Face dataset.
	"""
	dataset = load_dataset_dict(path, name, split)
	dataset = concatenate_datasets([
		dataset[name][split] for name in dataset for split in dataset[name]
	])

	if filter_fn is not None:
		dataset = dataset.filter(
			filter_fn,
			keep_in_memory=True,
			load_from_cache_file=False,
		)

	if process_fn is not None:
		dataset = dataset.map(
			process_fn,
			keep_in_memory=True,
			load_from_cache_file=False,
		)

	return dataset


def load_vllm_generator(
	path: str,
	trust_remote_code: bool = False,
	dtype: str = "bfloat16",
	tensor_parallel_size: int = 1,
) -> Any:
	"""Load a vLLM generator model instance."""
	try:
		LLM = importlib.import_module("vllm").LLM
	except Exception as exc:
		raise ImportError(
			"vLLM is required for `load_vllm_generator`. Install it with `pip install vllm`."
		) from exc

	path = resolve_model_path(path)
	return LLM(
		path,
		trust_remote_code=trust_remote_code,
		dtype=dtype,
		tensor_parallel_size=tensor_parallel_size,
	)


def load_hf_generator(
	path: str,
	trust_remote_code: bool = False,
	dtype: str = "bfloat16",
) -> AutoModelForCausalLM:
	"""Load a causal language model from Hugging Face."""
	path = resolve_model_path(path)
	dtype = resolve_torch_dtype(dtype)
	return AutoModelForCausalLM.from_pretrained(
		path,
		trust_remote_code=trust_remote_code,
		dtype=dtype,
	)


def load_hf_encoder(
	path: str,
	trust_remote_code: bool = False,
	dtype: str = "bfloat16",
) -> AutoModelForSequenceClassification:
	"""Load a sequence classification model from Hugging Face."""
	path = resolve_model_path(path)
	dtype = resolve_torch_dtype(dtype)
	return AutoModelForSequenceClassification.from_pretrained(
		path,
		num_labels=2,
		trust_remote_code=trust_remote_code,
		dtype=dtype,
	)


def load_hf_tokenizer(
	path: str,
	trust_remote_code: bool = False,
) -> AutoTokenizer:
	"""Load a tokenizer from Hugging Face."""
	path = resolve_model_path(path)
	return AutoTokenizer.from_pretrained(
		path,
		trust_remote_code=trust_remote_code,
	)


def resolve_dataset_path(path: str) -> str:
	"""Resolve dataset path from LOCAL_DATASETS_DIR when configured."""
	if not os.path.exists(path) and "LOCAL_DATASETS_DIR" in os.environ:
		path = os.path.join(
			os.environ["LOCAL_DATASETS_DIR"],
			path.split("/")[-1],
		)
	return path


def resolve_model_path(path: str) -> str:
	"""Resolve model path from LOCAL_MODELS_DIR when configured."""
	if not os.path.exists(path) and "LOCAL_MODELS_DIR" in os.environ:
		path = os.path.join(
			os.environ["LOCAL_MODELS_DIR"],
			path.split("/")[-1],
		)
	return path


def resolve_torch_dtype(dtype: str | torch.dtype) -> str | torch.dtype:
	"""Resolve string dtype to torch dtype or pass through accepted values."""
	if isinstance(dtype, torch.dtype):
		return dtype
	if dtype == "auto":
		return "auto"
	if isinstance(dtype, str) and hasattr(torch, dtype):
		return getattr(torch, dtype)
	raise ValueError(f"Unsupported dtype: {dtype}")


def parse_tasks(task_values: list[str]) -> list[str]:
	"""Parse mixed comma/space-separated task CLI values."""
	tasks = []
	for value in task_values:
		for task_name in value.split(","):
			task_name = task_name.strip()
			if task_name:
				tasks.append(task_name)
	return tasks


def discover_task_functions(
	package_name: str = "bert_judge.tasks",
) -> dict[str, Callable[..., Any]]:
	"""Discover callable task factory functions in a task package."""
	discovered: dict[str, Callable[..., Any]] = {}
	package = importlib.import_module(package_name)
	for module_info in pkgutil.iter_modules(package.__path__):
		if module_info.name.startswith("_"):
			continue

		module = importlib.import_module(f"{package_name}.{module_info.name}")
		for attr_name in dir(module):
			if attr_name.startswith("_"):
				continue
			attr_value = getattr(module, attr_name)
			if callable(attr_value) and getattr(attr_value, "__module__", None) == module.__name__:
				discovered[attr_name] = attr_value

	return discovered


def load_json_list(path: Path) -> list[Any]:
	"""Load and validate a JSON list from disk.

	Args:
		path: Path to a JSON file.

	Raises:
		FileNotFoundError: If the file does not exist.
		TypeError: If the JSON payload is not a list.
	"""
	if not path.exists():
		raise FileNotFoundError(f"File not found: {path}")

	data = json.loads(path.read_text(encoding="utf-8"))
	if not isinstance(data, list):
		raise TypeError(f"Expected a JSON list in {path}")

	return data


def get_model_name(model_path: str | Path) -> str:
	"""Return a normalized model directory name from a model path."""
	return str(model_path).rstrip("/").split("/")[-1].replace("-", "_")


def format_number(value: float | int) -> str:
	"""Format numeric values for stable path-friendly suffixes."""
	if isinstance(value, int):
		return str(value)
	return f"{value:g}".replace(".", "_")


def build_output_model_name(
	model_name: str,
	temperature: float = None,
	top_p: float = None,
	top_k: int = None,
	min_p: float = None,
	presence_penalty: float = None,
	max_tokens: int = None,
	enable_thinking: bool = None,
	instruction_type: str = None,
	metric: str = None,
) -> str:
	"""Build model-name suffix from generation configuration.

	Default deterministic setting keeps the original model name.
	"""
	output_model_name = model_name

	if enable_thinking:
		output_model_name += "_think"

	if temperature and temperature > 0:
		output_model_name += f"_t{format_number(temperature)}"
		output_model_name += f"_p{format_number(top_p)}"
		output_model_name += f"_k{format_number(top_k)}"
		output_model_name += f"_mp{format_number(min_p)}"
		output_model_name += f"_pp{format_number(presence_penalty)}"
		output_model_name += f"_mt{format_number(max_tokens)}"

	if instruction_type and instruction_type != "soft":
		output_model_name += f"_{instruction_type}"

	if metric:
		output_model_name += f"_{metric}"

	return output_model_name
