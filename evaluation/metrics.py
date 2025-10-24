"""
Data classes for evaluation metrics and results.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class RougeScores:
    """Container for ROUGE scores."""
    rouge1_precision: float
    rouge1_recall: float
    rouge1_fmeasure: float
    rouge2_precision: float
    rouge2_recall: float
    rouge2_fmeasure: float
    rougeL_precision: float
    rougeL_recall: float
    rougeL_fmeasure: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for easy serialization."""
        return {
            'rouge1_p': self.rouge1_precision,
            'rouge1_r': self.rouge1_recall,
            'rouge1_f': self.rouge1_fmeasure,
            'rouge2_p': self.rouge2_precision,
            'rouge2_r': self.rouge2_recall,
            'rouge2_f': self.rouge2_fmeasure,
            'rougeL_p': self.rougeL_precision,
            'rougeL_r': self.rougeL_recall,
            'rougeL_f': self.rougeL_fmeasure,
        }


@dataclass
class QAResult:
    """Container for QA evaluation results."""
    doc_id: str
    question: str
    reference_answer: str
    generated_answer: str
    accuracy_score: float  # 0-1 scale
    question_index: int
    paragraph_index: int
