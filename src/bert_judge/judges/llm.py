import re

from ..generators import (
    HFGenerator,
    vLLMGenerator,
)


class LLMJudge:
    """Judge candidates with an LLM instructed to return correctness labels."""

    def __init__(
        self,
        model_path: str,
        backend: str = "vllm",
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
        tensor_parallel_size: int = 1,
        device_map: str = "auto",
    ) -> None:
        """Initialize the LLM judge and backend generator."""
        self.prompt_template = (
            "You are an expert evaluator. Your task is to determine whether the "
            "CANDIDATE response correctly answers the QUESTION.\n"
            "Judge the CANDIDATE as correct only if its final answer, disregarding "
            "any intermediate reasoning or explanation, is semantically equivalent "
            "to the REFERENCE with respect to the QUESTION.\n"
            "Base your judgment solely on the information given. Do not rely on "
            "external knowledge.\n\n"
            "[QUESTION starts here]\n{question}\n[QUESTION ends here]\n\n"
            "[REFERENCE starts here]\n{reference}\n[REFERENCE ends here]\n\n"
            "[CANDIDATE starts here]\n{candidate}\n[CANDIDATE ends here]\n\n"
            "{instruction}:\n"
            '- "True" if the CANDIDATE is correct\n'
            '- "False" if the CANDIDATE is incorrect'
        )
        self.strict_instruction = (
            "Respond only with exactly one of the following strings (add no additional text)"
        )
        self.soft_instruction = "Conclude your response with exactly one of the following"

        if backend == "vllm":
            self.generator = vLLMGenerator(
                model_path=model_path,
                trust_remote_code=trust_remote_code,
                dtype=dtype,
                tensor_parallel_size=tensor_parallel_size,
                enable_thinking=enable_thinking,
                think_token=think_token,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                presence_penalty=presence_penalty,
                max_tokens=max_tokens,
            )
        elif backend == "hf":
            self.generator = HFGenerator(
                model_path=model_path,
                trust_remote_code=trust_remote_code,
                dtype=dtype,
                device_map=device_map,
                enable_thinking=enable_thinking,
                think_token=think_token,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                presence_penalty=presence_penalty,
                max_tokens=max_tokens,
            )
        else:
            raise ValueError(f"Unsupported inference backend: {backend}. Expected 'vllm' or 'hf'.")

    def predict(
        self,
        questions: list[str],
        candidates: list[str],
        references: list[str],
        instruction_type: str,
    ) -> list[int]:
        """Predict binary correctness scores for candidate answers."""
        prompts = self._apply_prompt_template(
            questions,
            candidates,
            references,
            instruction_type,
        )
        outputs = self.generator.generate(prompts)
        return self._compute_scores(outputs)

    def _apply_prompt_template(
        self,
        questions: list[str],
        candidates: list[str],
        references: list[str],
        instruction_type: str,
    ) -> list[str]:
        """Build judge prompts and select output parsing strategy."""
        if instruction_type == "strict":
            instruction = self.strict_instruction
            self._compute_scores = self._compute_scores_strict
        elif instruction_type == "soft":
            instruction = self.soft_instruction
            self._compute_scores = self._compute_scores_soft
        else:
            raise ValueError(
                f"Unsupported instruction_type: {instruction_type}. Expected 'strict' or 'soft'."
            )

        prompts = [
            self.prompt_template.format(
                question=question,
                candidate=candidate,
                reference=reference,
                instruction=instruction,
            )
            for question, candidate, reference in zip(
                questions, candidates, references, strict=False
            )
        ]
        return prompts

    def _compute_scores_strict(self, outputs: list[str]) -> list[int]:
        """Return 1 when output is exactly `True`, else 0."""
        return [(output == "True") * 1 for output in outputs]

    def _compute_scores_soft(self, outputs: list[str]) -> list[int]:
        """Extract final answer marker and map it to binary score."""
        scores: list[int] = []
        for output in outputs:
            match = re.findall(r"Final answer: (True|False)", output)
            score = (match[0] == "True") * 1 if len(match) > 0 else 0
            scores.append(score)
        return scores
