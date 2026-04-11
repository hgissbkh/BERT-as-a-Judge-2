import re
from string import ascii_uppercase

from math_verify import parse, verify
from rouge_score import rouge_scorer


class RegexJudge:
	"""Judge candidates using regex extraction and deterministic metrics."""

	def __init__(
		self,
		pattern: str = "Final answer:\\s*(.+)",
		metric: str = "EM",
	) -> None:
		"""Initialize extraction pattern and scoring metric."""
		self.pattern = pattern
		self.metric = metric

		if metric == "EM":
			self._compute_scores = self._compute_em_scores
		elif metric == "ROUGE":
			self._compute_scores = self._compute_rouge_scores
		elif metric == "Math-Verify":
			self._compute_scores = self._compute_math_verify_scores
		else:
			raise ValueError(
				f"Unsupported metric: {metric}. Expected one of ['EM', 'ROUGE', 'Math-Verify']."
			)

	def predict(
		self,
		candidates: list[str],
		references: list[str],
	) -> list[float | int]:
		"""Score candidate answers against references."""
		references = self._process_references(references)
		extractions = self._extract_answers(candidates)
		scores = self._compute_scores(extractions, references)
		return scores

	def _extract_answers(self, candidates: list[str]) -> list[str | None]:
		"""Extract answers from model outputs with configured regex."""
		extractions: list[str | None] = []
		for candidate in candidates:
			match = re.findall(self.pattern, candidate)
			extractions.append(match[0] if match else None)

		return extractions

	def _compute_em_scores(self, extractions: list[str | None], references: list[str]) -> list[int]:
		"""Compute exact-match binary scores."""
		return [
			(extraction == reference) * 1
			for extraction, reference in zip(extractions, references, strict=False)
		]

	def _compute_rouge_scores(
		self, extractions: list[str | None], references: list[str]
	) -> list[float]:
		"""Compute ROUGE-L F1 scores."""
		metric = rouge_scorer.RougeScorer(["rougeL"])
		scores: list[float] = []
		for extraction, reference in zip(extractions, references, strict=False):
			if extraction is None or reference is None:
				scores.append(0.0)
				continue

			scores.append(metric.score(reference, extraction)["rougeL"].fmeasure)

		return scores

	def _compute_math_verify_scores(
		self, extractions: list[str | None], references: list[str]
	) -> list[int]:
		"""Compute symbolic math verification scores."""
		scores: list[int] = []
		for extraction, reference in zip(extractions, references, strict=False):
			if extraction is None or reference is None:
				scores.append(0)
				continue

			scores.append(verify(parse(reference), parse(extraction)) * 1)

		return scores

	def _process_references(self, references: list[str]) -> list[str]:
		"""Normalize multiple-choice references to option labels when present."""
		return [
			reference[0] if reference and reference.split(")")[0] in ascii_uppercase else reference
			for reference in references
		]
