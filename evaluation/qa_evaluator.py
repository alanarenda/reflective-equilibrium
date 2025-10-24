"""
QA-based evaluation for summarization quality.

Uses an LM to answer questions from summaries and compares to reference answers.
"""

import re
import threading
from typing import List, Dict, Tuple
from together import Together
from evaluation.metrics import QAResult


class QAEvaluator:
    """
    Evaluates summarization quality using question-answering accuracy.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "Qwen/QwQ-32B-Preview"
    ):
        """
        Initialize the QA evaluator.

        Args:
            api_key: Together API key
            model: Model to use for QA evaluation
        """
        self.api_key = api_key
        self.model = model
        self._thread_local = threading.local()

    def _get_client(self) -> Together:
        """Get thread-local Together client."""
        if not hasattr(self._thread_local, 'client'):
            self._thread_local.client = Together(api_key=self.api_key)
        return self._thread_local.client

    def answer_question_from_summary(
        self,
        question: str,
        summary: str,
        max_tokens: int = 500
    ) -> str:
        """
        Use an LM to answer a question using only the provided summary.

        Args:
            question: The question to answer
            summary: The summary to use as context
            max_tokens: Maximum tokens for the answer

        Returns:
            The generated answer
        """
        prompt = f"""You are given a summary of a document and a question about it.
Answer the question using ONLY the information provided in the summary.
If the information needed to answer the question is not in the summary, say "Information not available in summary."

Summary:
{summary}

Question:
{question}

Answer:"""

        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.1,  # Low temperature for more deterministic answers
        )

        answer = response.choices[0].message.content.strip()
        return answer

    def evaluate_qa_accuracy(
        self,
        generated_answer: str,
        reference_answer: str
    ) -> float:
        """
        Evaluate QA accuracy by comparing generated answer to reference.

        Uses an LM to judge semantic similarity on a 0-1 scale.

        Args:
            generated_answer: Answer from the summary
            reference_answer: Ground truth answer

        Returns:
            Accuracy score from 0 to 1
        """
        prompt = f"""Compare the following two answers to determine how semantically similar they are.
The first answer is the reference (gold standard) answer.
The second answer is a generated answer that should be evaluated.

Rate the similarity on a scale from 0 to 1, where:
- 0 = Completely different, wrong information
- 0.5 = Partially correct, some information matches
- 1.0 = Semantically equivalent, captures the same meaning

Respond with ONLY a number between 0 and 1.

Reference Answer:
{reference_answer}

Generated Answer:
{generated_answer}

Similarity Score (0-1):"""

        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.0,
        )

        score_text = response.choices[0].message.content.strip()

        # Extract number from response
        match = re.search(r'([0-9]*\.?[0-9]+)', score_text)
        if match:
            score = float(match.group(1))
            return max(0.0, min(1.0, score))  # Clamp to [0, 1]
        else:
            # If parsing fails, return 0
            return 0.0

    def evaluate_document_qa(
        self,
        doc_id: str,
        generated_summary: str,
        questions: List[str],
        reference_answers: List[str],
        question_indices: List[int],
        paragraph_indices: List[int]
    ) -> Tuple[List[QAResult], float]:
        """
        Evaluate QA performance for all questions in a document.

        Args:
            doc_id: Document identifier
            generated_summary: The generated summary to evaluate
            questions: List of questions
            reference_answers: List of reference answers
            question_indices: Question indices in hierarchy
            paragraph_indices: Paragraph indices for each question

        Returns:
            Tuple of (list of QAResult objects, average accuracy score)
        """
        results = []

        for i, (question, ref_answer, q_idx, p_idx) in enumerate(
            zip(questions, reference_answers, question_indices, paragraph_indices)
        ):
            # Generate answer from summary
            generated_answer = self.answer_question_from_summary(question, generated_summary)

            # Evaluate accuracy
            accuracy = self.evaluate_qa_accuracy(generated_answer, ref_answer)

            results.append(QAResult(
                doc_id=doc_id,
                question=question,
                reference_answer=ref_answer,
                generated_answer=generated_answer,
                accuracy_score=accuracy,
                question_index=q_idx,
                paragraph_index=p_idx
            ))

        avg_accuracy = sum(r.accuracy_score for r in results) / len(results) if results else 0.0
        return results, avg_accuracy
