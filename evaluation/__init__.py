"""
Evaluation package for summarization quality assessment.

Provides ROUGE scoring and QA accuracy metrics.
"""

from evaluation.rouge_evaluator import RougeEvaluator
from evaluation.qa_evaluator import QAEvaluator
from evaluation.metrics import RougeScores, QAResult
from evaluation.evaluator import Evaluator

__all__ = [
    'Evaluator',
    'RougeEvaluator',
    'QAEvaluator',
    'RougeScores',
    'QAResult',
]
