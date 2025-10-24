"""Reflective summarization based on iterative surprise scoring"""

import re
from typing import List, Tuple
from .base import BaseSummarizer
from .utils import extract_summary, split_into_sentences


class ReflectiveSummarizer(BaseSummarizer):
    """
    Iterative summarization based on sentence-level surprise scoring.

    This method:
    1. Generates an initial summary
    2. Scores each sentence in the document for "surprise" given the current summary
    3. Refines the summary by incorporating high-surprise sentences
    4. Repeats for multiple iterations
    """

    def __init__(self,
                 surprise_threshold: float = 7.0,
                 max_iterations: int = 3,
                 top_k_sentences: int = 5,
                 **kwargs):
        """
        Initialize reflective summarizer.

        Args:
            surprise_threshold: Minimum surprise score (0-10) to consider a sentence
            max_iterations: Maximum number of refinement iterations
            top_k_sentences: Number of top surprising sentences to incorporate per iteration
            **kwargs: Additional configuration passed to parent
        """
        super().__init__(**kwargs)
        self.surprise_threshold = surprise_threshold
        self.max_iterations = max_iterations
        self.top_k_sentences = top_k_sentences

    def summarize(self, document: str, prompt: str) -> str:
        """
        Generate a summary using reflective method.

        Args:
            document: The full document text
            prompt: The summarization instruction/prompt

        Returns:
            The refined summary after iterative improvement
        """
        client = self.get_client()

        # Generate initial summary
        current_summary = self._generate_initial_summary(document, prompt, client)

        # Split document into sentences
        sentences = split_into_sentences(document)

        # Iteratively refine based on surprise
        for iteration in range(self.max_iterations):
            surprising_sentences = self._find_surprising_sentences(
                sentences, current_summary, prompt, client
            )

            if not surprising_sentences:
                # No more surprising sentences, we're done
                break

            # Incorporate top surprising sentences
            for sentence, score in surprising_sentences[:self.top_k_sentences]:
                current_summary = self._refine_summary(
                    current_summary, sentence, prompt, client
                )

        return current_summary

    def _generate_initial_summary(self, document: str, prompt: str, client) -> str:
        """
        Generate the initial summary before refinement.

        Args:
            document: The full document text
            prompt: The summarization instruction/prompt
            client: Together API client

        Returns:
            Initial summary text
        """
        full_prompt = f"{prompt}\n\nDocument: {document}"
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": full_prompt}],
            enable_thinking=True
        )
        return extract_summary(response.choices[0].message.content)

    def _compute_surprise_score(self, sentence: str, summary: str,
                                prompt: str, client) -> float:
        """
        Score how surprising/novel a sentence is given the current summary.

        Args:
            sentence: The sentence to score
            summary: Current summary text
            prompt: The summarization task/prompt
            client: Together API client

        Returns:
            Surprise score from 0-10
        """
        surprise_prompt = f"""Given this summarization task and current summary, rate how surprising/novel the following sentence is on a scale of 0-10.

Task: {prompt}

Current Summary: {summary}

Sentence to evaluate: {sentence}

Respond with just a number 0-10, where:
- 0 = completely covered by current summary
- 10 = highly surprising/important information not in summary

Score:"""

        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": surprise_prompt}],
            enable_thinking=False,
            max_tokens=10
        )

        try:
            score_text = response.choices[0].message.content.strip()
            score = float(re.search(r'\d+\.?\d*', score_text).group())
            return min(max(score, 0), 10)  # Clamp to 0-10 range
        except:
            return 0.0

    def _find_surprising_sentences(self, sentences: List[str], summary: str,
                                   prompt: str, client) -> List[Tuple[str, float]]:
        """
        Find sentences above surprise threshold, sorted by score.

        Args:
            sentences: List of all sentences in document
            summary: Current summary text
            prompt: The summarization task/prompt
            client: Together API client

        Returns:
            List of (sentence, score) tuples, sorted by score descending
        """
        scored = []
        for sentence in sentences:
            score = self._compute_surprise_score(sentence, summary, prompt, client)
            if score >= self.surprise_threshold:
                scored.append((sentence, score))

        return sorted(scored, key=lambda x: x[1], reverse=True)

    def _refine_summary(self, current_summary: str, surprising_sentence: str,
                       prompt: str, client) -> str:
        """
        Refine summary to incorporate a surprising sentence.

        Args:
            current_summary: The current summary text
            surprising_sentence: High-surprise sentence to incorporate
            prompt: The summarization task/prompt
            client: Together API client

        Returns:
            Updated summary incorporating the new information
        """
        refinement_prompt = f"""You have a current summary and a new important sentence to incorporate.

Task: {prompt}

Current Summary: {current_summary}

Important new information: {surprising_sentence}

Please provide an updated summary that incorporates this new information while staying concise.

Updated Summary:"""

        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": refinement_prompt}],
            enable_thinking=True
        )

        return extract_summary(response.choices[0].message.content)

    def name(self) -> str:
        """Return unique identifier for this method"""
        return f"reflective_t{self.surprise_threshold}_i{self.max_iterations}_k{self.top_k_sentences}"
