from ..utils import (
    load_vllm_generator,
)
from .base import BaseGenerator


class vLLMGenerator(BaseGenerator):
    """Generator implementation backed by vLLM."""

    def __init__(
        self,
        model_path: str,
        trust_remote_code: bool = False,
        dtype: str = "bfloat16",
        tensor_parallel_size: int = 1,
        enable_thinking: bool = False,
        think_token: str = "</think>",
        temperature: float = 0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
        presence_penalty: float = 0.0,
        max_tokens: int = 2048,
    ) -> None:
        """Initialize vLLM model and sampling parameters."""
        try:
            from vllm import SamplingParams
        except Exception as exc:
            raise ImportError(
                "vLLM is required for `vLLMGenerator`. Install it with `pip install vllm`."
            ) from exc

        super().__init__(
            model_path=model_path,
            trust_remote_code=trust_remote_code,
            dtype=dtype,
            enable_thinking=enable_thinking,
            think_token=think_token,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            presence_penalty=presence_penalty,
            max_tokens=max_tokens,
        )
        self.model = load_vllm_generator(
            model_path,
            trust_remote_code,
            dtype,
            tensor_parallel_size,
        )
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            presence_penalty=presence_penalty,
            max_tokens=max_tokens,
        )

    def generate(
        self,
        prompts: list[str],
    ) -> list[str]:
        """Generate one completion per prompt with prompt truncation."""
        prompts = self._apply_chat_template(prompts)
        prompts = self._truncate_prompts(
            prompts,
            self.model.llm_engine.model_config.max_model_len - self.max_tokens,
        )
        outputs = self.model.generate(prompts, self.sampling_params)
        outputs = [output.outputs[0].text.strip() for output in outputs]

        if self.enable_thinking:
            return self._extract_answers(outputs)

        return outputs

    def _truncate_prompts(
        self,
        prompts: list[str],
        max_prompt_tokens: int,
    ) -> list[str]:
        """Truncate prompts to fit the model context window."""
        max_prompt_tokens = max(1, int(max_prompt_tokens))
        truncated_prompts: list[str] = []

        for prompt in prompts:
            tokens = self.tokenizer(
                prompt,
                truncation=True,
                max_length=max_prompt_tokens,
            )["input_ids"]
            truncated_prompt = self.tokenizer.decode(tokens)
            truncated_prompts.append(truncated_prompt)

        return truncated_prompts
