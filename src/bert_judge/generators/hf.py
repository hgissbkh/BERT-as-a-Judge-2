import inspect
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorWithPadding

from ..utils import (
    load_hf_generator,
)
from .base import BaseGenerator


class HFGenerator(BaseGenerator):
    """Generator implementation backed by Hugging Face `generate`."""

    def __init__(
        self,
        model_path: str,
        trust_remote_code: bool = False,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        temperature: float = 0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
        presence_penalty: float = 0.0,
        max_tokens: int = 2048,
    ) -> None:
        """Initialize the Hugging Face text generator."""
        super().__init__(
            model_path,
            trust_remote_code,
            dtype,
            temperature,
            top_p,
            top_k,
            min_p,
            presence_penalty,
            max_tokens,
        )
        self.model = load_hf_generator(
            model_path,
            trust_remote_code,
            dtype,
            device_map,
        )
        self.max_prompt_tokens = max(1, self.model.config.max_position_embeddings - self.max_tokens)
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.eval()

    def generate(
        self,
        prompts: list[str],
        batch_size: int = 1,
    ) -> list[str]:
        """Generate one completion per prompt."""
        prompts = self._apply_chat_template(prompts)
        tokenized_prompts = self._tokenize_prompts(prompts)
        dataloader = self._build_dataloader(tokenized_prompts, batch_size)
        generation_kwargs = self._build_generation_kwargs()
        generated_texts = []

        with torch.no_grad():
            for batch in tqdm(dataloader):
                batch = {k: v.to(self.model.device) for k, v in batch.items()}
                outputs = self.model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    **generation_kwargs,
                )

                prompt_lengths = batch["attention_mask"].sum(dim=1).tolist()
                for index, prompt_length in enumerate(prompt_lengths):
                    response_ids = outputs[index, int(prompt_length) :]
                    generated_texts.append(
                        self.tokenizer.decode(response_ids, skip_special_tokens=True)
                    )

        return generated_texts

    def _tokenize_prompts(
        self,
        prompts: list[str],
    ) -> list[dict[str, Any]]:
        """Tokenize prompts with left truncation."""
        tokenized_prompts: list[dict[str, Any]] = []

        for prompt in prompts:
            tokenized_prompt = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_prompt_tokens,
            )
            tokenized_prompts.append(tokenized_prompt)

        return tokenized_prompts

    def _build_dataloader(
        self,
        tokenized_prompts: list[dict[str, Any]],
        batch_size: int = 1,
    ) -> DataLoader:
        """Build a deterministic dataloader for generation."""
        data_collator = DataCollatorWithPadding(self.tokenizer)
        return DataLoader(
            tokenized_prompts,
            batch_size=batch_size,
            collate_fn=data_collator,
            shuffle=False,
        )

    def _build_generation_kwargs(self) -> dict[str, Any]:
        """Prepare backend-compatible kwargs for `model.generate`."""
        generation_kwargs: dict[str, Any] = {
            "do_sample": self.temperature > 0,
            "max_new_tokens": self.max_tokens,
            "repetition_penalty": 1.0 + self.presence_penalty,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        if generation_kwargs["do_sample"]:
            generation_kwargs.update(
                {
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                }
            )

        generate_signature = inspect.signature(self.model.generate)
        if "min_p" in generate_signature.parameters:
            generation_kwargs["min_p"] = self.min_p

        return generation_kwargs
