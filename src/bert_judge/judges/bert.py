from typing import Any

import torch
from datasets import (
	Dataset,
	concatenate_datasets,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
	DataCollatorWithPadding,
	Trainer,
	TrainingArguments,
)

from ..utils import (
	get_model_name,
	load_hf_encoder,
	load_hf_tokenizer,
)


class BERTJudge:
	"""Encoder-based judge trained for reference-based candidate evaluation."""

	def __init__(
		self,
		model_path: str,
		trust_remote_code: bool = False,
		dtype: str = "bfloat16",
	) -> None:
		"""Initialize model, tokenizer, and required special tokens."""
		self.special_tokens = {
			"question": "<|question|>",
			"candidate": "<|candidate|>",
			"reference": "<|reference|>",
		}
		self.model = load_hf_encoder(
			model_path,
			trust_remote_code,
			dtype,
		)
		self.max_length = self.model.config.max_position_embeddings
		self.tokenizer = load_hf_tokenizer(
			model_path,
			trust_remote_code=trust_remote_code,
		)
		self.tokenizer.truncation_side = "left"
		self._add_special_tokens()

	def fit(
		self,
		dataset: dict[str, Any],
		output_dir: str,
		include_question: bool = True,
		training_mix: dict[str, dict[str, int]] | None = None,
		num_train_epochs: float = 1,
		batch_size: int = 4,
		learning_rate: float = 2e-5,
		warmup_ratio: float = 0.05,
		lr_scheduler_type: str = "linear",
		logging_strategy: str = "steps",
		logging_steps: int = 10,
		logging_dir: str | None = None,
		report_to: list[str] | None = None,
		save_strategy: str = "steps",
		save_steps: int = 500,
		save_total_limit: int = 1,
		seed: int = 0,
	) -> None:
		"""Train the BERT judge on prepared labeled datasets."""
		if report_to is None:
			report_to = ["tensorboard"]

		if training_mix:
			dataset = self._apply_training_mix(dataset, training_mix, seed)
		else:
			dataset = self._flatten_dataset(dataset)

		dataset = self._make_prompts(dataset, include_question)
		dataset = self._tokenize_prompts(dataset)
		trainer = self._build_trainer(
			dataset,
			output_dir,
			num_train_epochs,
			batch_size,
			learning_rate,
			warmup_ratio,
			lr_scheduler_type,
			logging_strategy,
			logging_steps,
			logging_dir,
			report_to,
			save_strategy,
			save_steps,
			save_total_limit,
			seed,
		)
		trainer.train()
		self._save_model(output_dir)

	def predict(
		self,
		questions: list[str],
		candidates: list[str],
		references: list[str],
		batch_size: int = 1,
	) -> list[float]:
		"""Predict per-example correctness probabilities."""
		if not questions:
			questions = [""] * len(references)
			include_question = False
		else:
			include_question = True

		dataset = Dataset.from_dict(
			{
				"question": questions,
				"candidate": candidates,
				"reference": references,
			}
		)
		dataset = self._make_prompts(dataset, include_question)
		dataset = self._tokenize_prompts(dataset)
		dataloader = self._build_dataloader(dataset, batch_size)

		self.model.eval()
		if torch.cuda.is_available():
			self.model = self.model.to("cuda")

		scores: list[list[float]] = []
		with torch.no_grad():
			for batch in tqdm(dataloader, "Computing scores"):
				batch = {k: v.to(self.model.device) for k, v in batch.items()}
				output = self.model(**batch)
				scores += output.logits.cpu().tolist()

		return [torch.tensor(s[1] - s[0]).sigmoid().item() for s in scores]

	def _add_special_tokens(self) -> None:
		"""Ensure tokenizer/model are aligned on padding and special tokens."""
		if self.tokenizer.pad_token_id is None:
			self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
		if self.model.config.pad_token_id is None:
			self.model.config.pad_token_id = self.tokenizer.pad_token_id

		self.tokenizer.add_tokens(list(self.special_tokens.values()))
		self.model.resize_token_embeddings(len(self.tokenizer))

	def _apply_training_mix(
		self,
		dataset: dict[str, Any],
		training_mix: dict[str, dict[str, int]],
		seed: int = 0,
	) -> Dataset:
		"""Sample and concatenate subsets according to `training_mix`."""
		processed_dataset: list[Dataset] = []
		for task_name in training_mix:
			for model_name, num_samples in training_mix[task_name].items():
				subset = dataset[task_name][get_model_name(model_name)].shuffle(seed)
				processed_dataset.append(subset.select(range(min(len(subset), num_samples))))

		return concatenate_datasets(processed_dataset)

	def _flatten_dataset(
		self,
		dataset: dict[str, Any],
	) -> Dataset:
		"""Flatten nested task/split datasets into a single train dataset."""
		return concatenate_datasets(
			[dataset[name][split] for name in dataset for split in dataset[name]]
		)

	def _make_prompts(
		self,
		dataset: Dataset,
		include_question: bool = True,
	) -> Dataset:
		"""Build model input prompts from question/candidate/reference fields."""

		def fn(ex):
			prompt = ""
			if include_question:
				prompt += self.special_tokens["question"] + ex["question"]
			prompt += self.special_tokens["candidate"] + ex["candidate"]
			prompt += self.special_tokens["reference"] + ex["reference"]
			return {"prompt": prompt}

		dataset = dataset.map(
			fn,
			keep_in_memory=True,
			load_from_cache_file=False,
		)

		if "label" in dataset.column_names:
			return dataset.select_columns(["prompt", "label"])
		else:
			return dataset.select_columns(["prompt"])

	def _tokenize_prompts(
		self,
		dataset: Dataset,
	) -> Dataset:
		"""Tokenize prompt field for model consumption."""

		def fn(ex):
			return self.tokenizer(
				ex["prompt"],
				truncation=True,
				max_length=self.max_length,
			)

		return dataset.map(
			fn,
			keep_in_memory=True,
			load_from_cache_file=False,
		).remove_columns(["prompt"])

	def _build_trainer(
		self,
		dataset: Dataset,
		output_dir: str,
		num_train_epochs: float = 1,
		batch_size: int = 4,
		learning_rate: float = 2e-5,
		warmup_ratio: float = 0.05,
		lr_scheduler_type: str = "linear",
		logging_strategy: str = "steps",
		logging_steps: int = 10,
		logging_dir: str | None = None,
		report_to: list[str] | None = None,
		save_strategy: str = "steps",
		save_steps: int = 500,
		save_total_limit: int = 1,
		seed: int = 0,
	) -> Trainer:
		"""Create the Hugging Face trainer with all training arguments."""
		if report_to is None:
			report_to = ["tensorboard"]

		training_args = TrainingArguments(
			num_train_epochs=num_train_epochs,
			per_device_train_batch_size=batch_size,
			per_device_eval_batch_size=batch_size,
			learning_rate=learning_rate,
			warmup_ratio=warmup_ratio,
			lr_scheduler_type=lr_scheduler_type,
			logging_strategy=logging_strategy,
			logging_steps=logging_steps,
			logging_dir=logging_dir,
			report_to=report_to,
			save_strategy=save_strategy,
			save_steps=save_steps,
			save_total_limit=save_total_limit,
			output_dir=output_dir,
			seed=seed,
		)
		data_collator = DataCollatorWithPadding(self.tokenizer)
		return Trainer(
			model=self.model,
			args=training_args,
			train_dataset=dataset,
			data_collator=data_collator,
		)

	def _save_model(
		self,
		output_dir: str,
	) -> None:
		"""Persist model and tokenizer to output directory."""
		self.model.save_pretrained(output_dir)
		self.tokenizer.save_pretrained(output_dir)

	def _build_dataloader(
		self,
		dataset: Dataset,
		batch_size: int,
	) -> DataLoader:
		"""Create deterministic dataloader for prediction."""
		data_collator = DataCollatorWithPadding(self.tokenizer)
		return DataLoader(
			dataset,
			batch_size=batch_size,
			collate_fn=data_collator,
			shuffle=False,
		)
