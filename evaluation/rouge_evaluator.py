"""
ROUGE-based evaluation for summarization quality.
"""

from typing import List
from rouge_score import rouge_scorer
from evaluation.metrics import RougeScores


class RougeEvaluator:
    """
    Evaluates summarization quality using ROUGE metrics.
    """

    def __init__(self, rouge_types: List[str] = ['rouge1', 'rouge2', 'rougeL']):
        """
        Initialize the ROUGE evaluator.

        Args:
            rouge_types: ROUGE metric types to compute (default: rouge1, rouge2, rougeL)
        """
        self.rouge_scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)

    def compute_rouge_scores(
        self,
        generated_summary: str,
        reference_summary: str
    ) -> RougeScores:
        """
        Compute ROUGE scores between generated and reference summaries.

        Args:
            generated_summary: The generated summary text
            reference_summary: The reference (gold) summary text

        Returns:
            RougeScores object with precision, recall, and F-measure for each ROUGE type
        """
        scores = self.rouge_scorer.score(reference_summary, generated_summary)

        return RougeScores(
            rouge1_precision=scores['rouge1'].precision,
            rouge1_recall=scores['rouge1'].recall,
            rouge1_fmeasure=scores['rouge1'].fmeasure,
            rouge2_precision=scores['rouge2'].precision,
            rouge2_recall=scores['rouge2'].recall,
            rouge2_fmeasure=scores['rouge2'].fmeasure,
            rougeL_precision=scores['rougeL'].precision,
            rougeL_recall=scores['rougeL'].recall,
            rougeL_fmeasure=scores['rougeL'].fmeasure,
        )
