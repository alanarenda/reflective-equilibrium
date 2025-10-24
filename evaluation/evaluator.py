"""
Main evaluator that combines ROUGE and QA evaluation.
"""

from typing import Dict, List, Optional
from evaluation.rouge_evaluator import RougeEvaluator
from evaluation.qa_evaluator import QAEvaluator
from evaluation.metrics import RougeScores, QAResult


class Evaluator:
    """
    Comprehensive evaluator combining ROUGE scores and QA accuracy.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "Qwen/QwQ-32B-Preview",
        rouge_types: List[str] = ['rouge1', 'rouge2', 'rougeL']
    ):
        """
        Initialize the evaluator.

        Args:
            api_key: Together API key (for QA evaluation)
            model: Model to use for QA evaluation
            rouge_types: ROUGE metric types to compute
        """
        self.rouge_evaluator = RougeEvaluator(rouge_types=rouge_types)
        self.qa_evaluator = QAEvaluator(api_key=api_key, model=model)

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
            RougeScores object with precision, recall, and F-measure
        """
        return self.rouge_evaluator.compute_rouge_scores(
            generated_summary,
            reference_summary
        )

    def evaluate_summary(
        self,
        doc_id: str,
        generated_summary: str,
        reference_summary: str,
        hierarchy: Optional[Dict] = None,
        compute_qa: bool = True
    ) -> Dict:
        """
        Comprehensive evaluation of a single summary.

        Args:
            doc_id: Document identifier
            generated_summary: Generated summary to evaluate
            reference_summary: Reference summary for ROUGE
            hierarchy: Question hierarchy for QA evaluation
            compute_qa: Whether to compute QA metrics

        Returns:
            Dictionary with evaluation results
        """
        results = {
            'doc_id': doc_id,
            'rouge_scores': None,
            'qa_results': None,
            'avg_qa_accuracy': None
        }

        # Compute ROUGE scores
        rouge_scores = self.compute_rouge_scores(generated_summary, reference_summary)
        results['rouge_scores'] = rouge_scores.to_dict()

        # Compute QA accuracy if hierarchy provided
        if compute_qa and hierarchy:
            questions = self._extract_questions_from_hierarchy(hierarchy)
            reference_answers = self._extract_answers_from_hierarchy(hierarchy)
            question_indices = self._extract_question_indices(hierarchy)
            paragraph_indices = self._extract_paragraph_indices(hierarchy)

            qa_results, avg_accuracy = self.qa_evaluator.evaluate_document_qa(
                doc_id,
                generated_summary,
                questions,
                reference_answers,
                question_indices,
                paragraph_indices
            )

            results['qa_results'] = [
                {
                    'question': r.question,
                    'reference_answer': r.reference_answer,
                    'generated_answer': r.generated_answer,
                    'accuracy': r.accuracy_score,
                    'question_index': r.question_index,
                    'paragraph_index': r.paragraph_index
                }
                for r in qa_results
            ]
            results['avg_qa_accuracy'] = avg_accuracy

        return results

    def _extract_questions_from_hierarchy(self, hierarchy: Dict) -> List[str]:
        """Extract all questions from nested hierarchy."""
        questions = []

        def traverse(node):
            if isinstance(node, dict):
                if 'question' in node:
                    questions.append(node['question'])
                if 'children' in node:
                    for child in node['children']:
                        traverse(child)
            elif isinstance(node, list):
                for item in node:
                    traverse(item)

        traverse(hierarchy)
        return questions

    def _extract_answers_from_hierarchy(self, hierarchy: Dict) -> List[str]:
        """Extract all reference answers from nested hierarchy."""
        answers = []

        def traverse(node):
            if isinstance(node, dict):
                if 'summary' in node:
                    answers.append(node['summary'])
                if 'children' in node:
                    for child in node['children']:
                        traverse(child)
            elif isinstance(node, list):
                for item in node:
                    traverse(item)

        traverse(hierarchy)
        return answers

    def _extract_question_indices(self, hierarchy: Dict) -> List[int]:
        """Extract question indices from nested hierarchy."""
        indices = []

        def traverse(node):
            if isinstance(node, dict):
                if 'index' in node:
                    indices.append(node['index'])
                if 'children' in node:
                    for child in node['children']:
                        traverse(child)
            elif isinstance(node, list):
                for item in node:
                    traverse(item)

        traverse(hierarchy)
        return indices

    def _extract_paragraph_indices(self, hierarchy: Dict) -> List[int]:
        """Extract paragraph indices from nested hierarchy."""
        indices = []

        def traverse(node):
            if isinstance(node, dict):
                if 'paragraph' in node:
                    indices.append(node['paragraph'])
                if 'children' in node:
                    for child in node['children']:
                        traverse(child)
            elif isinstance(node, list):
                for item in node:
                    traverse(item)

        traverse(hierarchy)
        return indices

    @staticmethod
    def aggregate_results(results_list: List[Dict]) -> Dict:
        """
        Aggregate evaluation results across multiple documents.

        Args:
            results_list: List of evaluation result dictionaries

        Returns:
            Dictionary with aggregated statistics
        """
        # Aggregate ROUGE scores
        rouge_scores = {
            'rouge1_f': [],
            'rouge2_f': [],
            'rougeL_f': [],
        }

        qa_accuracies = []

        for result in results_list:
            if result.get('rouge_scores'):
                rouge_scores['rouge1_f'].append(result['rouge_scores']['rouge1_f'])
                rouge_scores['rouge2_f'].append(result['rouge_scores']['rouge2_f'])
                rouge_scores['rougeL_f'].append(result['rouge_scores']['rougeL_f'])

            if result.get('avg_qa_accuracy') is not None:
                qa_accuracies.append(result['avg_qa_accuracy'])

        aggregated = {
            'num_documents': len(results_list),
            'rouge1_f_mean': sum(rouge_scores['rouge1_f']) / len(rouge_scores['rouge1_f']) if rouge_scores['rouge1_f'] else 0,
            'rouge2_f_mean': sum(rouge_scores['rouge2_f']) / len(rouge_scores['rouge2_f']) if rouge_scores['rouge2_f'] else 0,
            'rougeL_f_mean': sum(rouge_scores['rougeL_f']) / len(rouge_scores['rougeL_f']) if rouge_scores['rougeL_f'] else 0,
            'qa_accuracy_mean': sum(qa_accuracies) / len(qa_accuracies) if qa_accuracies else None,
        }

        return aggregated
