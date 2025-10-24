"""
Experiment runner for evaluating generated summaries.

This script evaluates summaries using:
1. ROUGE scores against reference summaries
2. QA accuracy by having an LM answer questions from the summary
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional
import threading
from tqdm import tqdm
import argparse

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from evaluation import Evaluator


class EvaluationExperiment:
    """
    Orchestrates evaluation experiments with checkpointing and parallel processing.
    """

    def __init__(
        self,
        api_key: str,
        data_dir: str = "data",
        output_dir: str = "evaluation_results",
        max_workers: int = 5,  # Fewer workers for evaluation to avoid rate limits
        checkpoint_every: int = 10,
        model: str = "Qwen/QwQ-32B-Preview"
    ):
        """
        Initialize evaluation experiment.

        Args:
            api_key: Together API key
            data_dir: Directory containing dataset files
            output_dir: Directory to save results
            max_workers: Number of parallel workers
            checkpoint_every: Save results every N documents
            model: Model to use for QA evaluation
        """
        self.api_key = api_key
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.max_workers = max_workers
        self.checkpoint_every = checkpoint_every
        self.model = model

        # Create output directory
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Thread lock for writing
        self.write_lock = threading.Lock()

        # Initialize evaluator
        self.evaluator = Evaluator(api_key=api_key, model=model)

    def load_dataset(self, split: str = "test") -> pd.DataFrame:
        """
        Load dataset split.

        Args:
            split: One of 'train', 'val', or 'test'

        Returns:
            DataFrame with document data
        """
        if split == "train":
            file_path = self.data_dir / "train_combined_with_prompts.csv"
        elif split == "val":
            file_path = self.data_dir / "val_combined.csv"
        elif split == "test":
            file_path = self.data_dir / "test_combined.csv"
        else:
            raise ValueError(f"Invalid split: {split}")

        df = pd.read_csv(file_path)
        return df

    def load_hierarchies(self) -> Dict:
        """Load question hierarchies."""
        hierarchy_path = self.data_dir / "question_hierarchies_nested.json"
        with open(hierarchy_path, 'r') as f:
            hierarchies = json.load(f)
        return hierarchies

    def load_generated_summaries(self, summary_file: str) -> pd.DataFrame:
        """
        Load generated summaries from CSV.

        Args:
            summary_file: Path to CSV file with generated summaries

        Returns:
            DataFrame with doc_id as index and summary column
        """
        df = pd.read_csv(summary_file, index_col=0)
        return df

    def evaluate_single_document(
        self,
        doc_id: str,
        generated_summary: str,
        reference_summary: str,
        hierarchy: Optional[Dict] = None,
        compute_qa: bool = True
    ) -> Dict:
        """
        Evaluate a single document.

        Args:
            doc_id: Document ID
            generated_summary: Generated summary
            reference_summary: Reference summary
            hierarchy: Question hierarchy
            compute_qa: Whether to compute QA metrics

        Returns:
            Evaluation results dictionary
        """
        try:
            results = self.evaluator.evaluate_summary(
                doc_id=doc_id,
                generated_summary=generated_summary,
                reference_summary=reference_summary,
                hierarchy=hierarchy,
                compute_qa=compute_qa
            )
            results['success'] = True
            return results

        except Exception as e:
            print(f"Error evaluating {doc_id}: {e}")
            return {
                'doc_id': doc_id,
                'success': False,
                'error': str(e)
            }

    def run_evaluation(
        self,
        summary_file: str,
        split: str = "test",
        compute_qa: bool = True,
        max_documents: Optional[int] = None
    ) -> str:
        """
        Run evaluation on generated summaries.

        Args:
            summary_file: Path to CSV with generated summaries
            split: Dataset split to use for reference summaries
            compute_qa: Whether to compute QA metrics
            max_documents: Maximum number of documents to evaluate (for testing)

        Returns:
            Path to results file
        """
        print(f"Loading dataset split: {split}")
        dataset = self.load_dataset(split)

        print(f"Loading generated summaries from: {summary_file}")
        generated_summaries = self.load_generated_summaries(summary_file)

        # Load hierarchies if computing QA
        hierarchies = None
        if compute_qa:
            print("Loading question hierarchies...")
            hierarchies = self.load_hierarchies()

        # Find common documents
        common_doc_ids = set(dataset['doc_id']).intersection(set(generated_summaries.index))
        print(f"Found {len(common_doc_ids)} documents to evaluate")

        if max_documents:
            common_doc_ids = list(common_doc_ids)[:max_documents]
            print(f"Limiting to {max_documents} documents")

        # Prepare tasks
        tasks = []
        for doc_id in common_doc_ids:
            # Get reference summary
            ref_row = dataset[dataset['doc_id'] == doc_id].iloc[0]
            reference_summary = ref_row['summary']

            # Get generated summary
            generated_summary = generated_summaries.loc[doc_id, 'summary']

            # Get hierarchy if available
            hierarchy = hierarchies.get(doc_id) if hierarchies else None

            tasks.append({
                'doc_id': doc_id,
                'generated_summary': generated_summary,
                'reference_summary': reference_summary,
                'hierarchy': hierarchy,
                'compute_qa': compute_qa
            })

        # Run evaluation with parallel processing
        results = []
        completed = 0

        print(f"Starting evaluation with {self.max_workers} workers...")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(
                    self.evaluate_single_document,
                    **task
                ): task for task in tasks
            }

            # Process completed tasks with progress bar
            with tqdm(total=len(tasks), desc="Evaluating") as pbar:
                for future in as_completed(future_to_task):
                    result = future.result()
                    results.append(result)
                    completed += 1

                    # Checkpoint results
                    if completed % self.checkpoint_every == 0:
                        self._save_results(results, summary_file, split, checkpoint=True)

                    pbar.update(1)

        # Save final results
        output_path = self._save_results(results, summary_file, split, checkpoint=False)

        # Print summary statistics
        self._print_summary(results)

        return output_path

    def _save_results(
        self,
        results: List[Dict],
        summary_file: str,
        split: str,
        checkpoint: bool = False
    ) -> str:
        """Save evaluation results to disk."""
        with self.write_lock:
            # Generate output filename
            summary_name = Path(summary_file).stem
            suffix = "_checkpoint" if checkpoint else ""
            output_name = f"{summary_name}_{split}_evaluation{suffix}.json"
            output_path = self.output_dir / output_name

            # Save results as JSON
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)

            # Also save aggregated statistics
            if not checkpoint:
                agg_results = self.evaluator.aggregate_results(
                    [r for r in results if r.get('success', False)]
                )

                stats_name = f"{summary_name}_{split}_stats.json"
                stats_path = self.output_dir / stats_name

                with open(stats_path, 'w') as f:
                    json.dump(agg_results, f, indent=2)

                # Save detailed results as CSV for easier analysis
                self._save_detailed_csv(results, summary_name, split)

        return str(output_path)

    def _save_detailed_csv(
        self,
        results: List[Dict],
        summary_name: str,
        split: str
    ):
        """Save detailed results as CSV."""
        # Extract ROUGE scores
        rouge_data = []
        for result in results:
            if result.get('success') and result.get('rouge_scores'):
                row = {
                    'doc_id': result['doc_id'],
                    **result['rouge_scores'],
                    'avg_qa_accuracy': result.get('avg_qa_accuracy')
                }
                rouge_data.append(row)

        rouge_df = pd.DataFrame(rouge_data)
        rouge_csv = self.output_dir / f"{summary_name}_{split}_rouge_scores.csv"
        rouge_df.to_csv(rouge_csv, index=False)

        # Extract QA results
        qa_data = []
        for result in results:
            if result.get('success') and result.get('qa_results'):
                for qa in result['qa_results']:
                    row = {
                        'doc_id': result['doc_id'],
                        **qa
                    }
                    qa_data.append(row)

        if qa_data:
            qa_df = pd.DataFrame(qa_data)
            qa_csv = self.output_dir / f"{summary_name}_{split}_qa_results.csv"
            qa_df.to_csv(qa_csv, index=False)

    def _print_summary(self, results: List[Dict]):
        """Print summary statistics."""
        successful = [r for r in results if r.get('success', False)]
        failed = [r for r in results if not r.get('success', False)]

        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Total documents: {len(results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")

        if successful:
            agg = self.evaluator.aggregate_results(successful)
            print(f"\nROUGE Scores (F-measure):")
            print(f"  ROUGE-1: {agg['rouge1_f_mean']:.4f}")
            print(f"  ROUGE-2: {agg['rouge2_f_mean']:.4f}")
            print(f"  ROUGE-L: {agg['rougeL_f_mean']:.4f}")

            if agg['qa_accuracy_mean'] is not None:
                print(f"\nQA Accuracy: {agg['qa_accuracy_mean']:.4f}")

        print("="*60 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate generated summaries using ROUGE and QA accuracy"
    )
    parser.add_argument(
        "summary_file",
        type=str,
        help="Path to CSV file with generated summaries"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to use for reference summaries (default: test)"
    )
    parser.add_argument(
        "--no-qa",
        action="store_true",
        help="Skip QA evaluation (only compute ROUGE)"
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Maximum number of documents to evaluate (for testing)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Number of parallel workers (default: 5)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/QwQ-32B-Preview",
        help="Model to use for QA evaluation"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Together API key (or set TOGETHER_API_KEY env var)"
    )

    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        raise ValueError("API key must be provided via --api-key or TOGETHER_API_KEY env var")

    # Initialize experiment
    experiment = EvaluationExperiment(
        api_key=api_key,
        max_workers=args.workers,
        model=args.model
    )

    # Run evaluation
    output_path = experiment.run_evaluation(
        summary_file=args.summary_file,
        split=args.split,
        compute_qa=not args.no_qa,
        max_documents=args.max_docs
    )

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
