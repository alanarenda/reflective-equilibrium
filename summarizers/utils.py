"""Utility functions for summarization"""

import re
from typing import List


def extract_summary(response_text: str) -> str:
    """
    Extract the actual summary from response, removing thinking tags if present.

    Args:
        response_text: Raw response text from the model

    Returns:
        Cleaned summary text
    """
    if response_text is None:
        return ""

    # Remove thinking tags and content
    # Pattern: <think>...</think> at the start
    cleaned = re.sub(r'<think>.*?</think>\s*', '', response_text, flags=re.DOTALL)
    return cleaned.strip()


def split_into_sentences(document: str) -> List[str]:
    """
    Split document into sentences using simple heuristic.

    Note: For production use, consider using nltk.sent_tokenize for better results.

    Args:
        document: Full document text

    Returns:
        List of sentences
    """
    sentences = re.split(r'(?<=[.!?])\s+', document)
    return [s.strip() for s in sentences if s.strip()]
