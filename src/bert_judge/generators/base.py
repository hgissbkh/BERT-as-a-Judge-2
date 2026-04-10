from ..utils import load_hf_tokenizer


class BaseGenerator:
    """Common generator base with tokenizer setup and chat formatting."""

    def __init__(
        self,
        model_path: str,
        trust_remote_code: bool = False,
        dtype: str = "bfloat16",
        enable_thinking: bool = False,
        think_token: str = "</think>",
        temperature: float = 0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
        presence_penalty: float = 0.0,
        max_tokens: int = 2048,
    ) -> None:
        """Initialize common generator configuration and tokenizer."""
        self.model_path = model_path
        self.trust_remote_code = trust_remote_code
        self.dtype = dtype
        self.enable_thinking = enable_thinking
        self.think_token = think_token
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.presence_penalty = presence_penalty
        self.max_tokens = max_tokens
        self.tokenizer = load_hf_tokenizer(
            model_path,
            trust_remote_code=trust_remote_code,
        )
        self.tokenizer.truncation_side = "left"
        self._configure_tokenizer_padding()

    def _configure_tokenizer_padding(self) -> None:
        """Set tokenizer padding token when missing."""
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _apply_chat_template(
        self,
        prompts: list[str],
    ) -> list[str]:
        """Apply tokenizer chat template when available."""
        if not hasattr(self.tokenizer, "apply_chat_template"):
            return prompts

        if self.model_path.split("/")[-1] == "Llama-3_3-Nemotron-Super-49B-v1_5":
            messages = [
                [
                    {"role": "system", "content": "/no_think"},
                    {"role": "user", "content": prompt},
                ]
                for prompt in prompts
            ]
        else:
            messages = [[{"role": "user", "content": prompt}] for prompt in prompts]

        processed_prompts: list[str] = []

        for _messages in messages:
            processed_prompt = self.tokenizer.apply_chat_template(
                _messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
            processed_prompts.append(processed_prompt)

        return processed_prompts

    def _extract_answers(
        self,
        outputs: list[str],
    ) -> list[str]:
        """Extract final answers after the configured thinking token."""
        chat_template = getattr(self.tokenizer, "chat_template", None)
        if not isinstance(chat_template, str) or self.think_token not in chat_template:
            raise ValueError(
                f"Configured think token `{self.think_token}` is not present in tokenizer chat template."
            )

        answers: list[str] = []
        for output in outputs:
            if self.think_token in output:
                answers.append(output.split(self.think_token, 1)[-1].strip())
            else:
                answers.append(output.strip())

        return answers
