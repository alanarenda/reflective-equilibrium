"""Orchestration for running summarization experiments"""

import os
import threading
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from summarizers.base import BaseSummarizer


class SummarizationExperiment:
    """Orchestrates running multiple summarization methods on a dataset"""

    def __init__(self,
                 summarizers: List[BaseSummarizer],
                 output_dir: str = 'data',
                 max_workers: int = 10,
                 checkpoint_every: int = 5):
        """
        Initialize the experiment.

        Args:
            summarizers: List of summarization methods to test
            output_dir: Directory to save results
            max_workers: Number of parallel workers for processing
            checkpoint_every: Save checkpoint every N successful results
        """
        self.summarizers = summarizers
        self.output_dir = output_dir
        self.max_workers = max_workers
        self.checkpoint_every = checkpoint_every
        self.write_lock = threading.Lock()

    def run(self, input_path: str, prompt_type: str = 'topic'):
        """
        Run all summarizers on the dataset.

        Args:
            input_path: Path to input CSV with documents
            prompt_type: 'base' or 'topic' - which prompt to use
        """
        df = pd.read_csv(input_path)

        for summarizer in self.summarizers:
            print(f"\n{'='*60}")
            print(f"Running: {summarizer.name()}")
            print(f"{'='*60}\n")

            self._run_single_summarizer(df, summarizer, prompt_type)

    def _run_single_summarizer(self, df: pd.DataFrame,
                               summarizer: BaseSummarizer,
                               prompt_type: str):
        """
        Run a single summarizer on the dataset.

        Args:
            df: DataFrame with documents
            summarizer: The summarization method to use
            prompt_type: 'base' or 'topic' prompt type
        """
        output_path = os.path.join(
            self.output_dir,
            f'{summarizer.name()}_{prompt_type}_summaries.csv'
        )

        # Load existing results
        processed_ids = self._load_existing_results(output_path)

        # Filter unprocessed rows
        rows_to_process = [
            (idx, row) for idx, row in df.iterrows()
            if row['id'] not in processed_ids
        ]

        print(f"Processing {len(rows_to_process)} documents "
              f"(out of {len(df)} total)")

        if len(rows_to_process) == 0:
            print("All documents already processed!")
            return

        # Process in parallel
        all_results = {}

        def process_row(row):
            """Process a single row to generate summary"""
            try:
                document = row['document']
                prompt = self._get_prompt(row, prompt_type)

                summary = summarizer.summarize(document, prompt)

                return {
                    'id': row['id'],
                    'summary': summary,
                    'success': True
                }
            except Exception as e:
                return {
                    'id': row['id'],
                    'error': str(e),
                    'success': False
                }

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(process_row, row): (idx, row['id'])
                for idx, row in rows_to_process
            }

            success_count = 0
            error_count = 0

            for future in tqdm(as_completed(futures),
                             total=len(futures),
                             desc=f"Generating summaries"):
                result = future.result()

                if result['success']:
                    all_results[result['id']] = {'summary': result['summary']}
                    success_count += 1

                    # Checkpoint
                    if success_count % self.checkpoint_every == 0:
                        self._save_results(all_results, output_path)
                        print(f"\n✓ Checkpoint: Saved {len(all_results)} results")
                else:
                    print(f"\nError processing {result['id']}: {result['error']}")
                    error_count += 1

        # Final save
        self._save_results(all_results, output_path)
        print(f"\n✓ Completed: {success_count} successful, {error_count} errors")
        print(f"✓ Results saved to {output_path}")

    def _get_prompt(self, row, prompt_type: str) -> str:
        """
        Get the appropriate prompt based on type.

        Args:
            row: DataFrame row
            prompt_type: 'base' or 'topic'

        Returns:
            The prompt string to use
        """
        if prompt_type == 'topic':
            return row['summarization_prompt']
        else:
            return "Summarize the following document:"

    def _load_existing_results(self, output_path: str) -> set:
        """
        Load already processed document IDs.

        Args:
            output_path: Path to existing results CSV

        Returns:
            Set of processed document IDs
        """
        if not os.path.exists(output_path):
            return set()

        try:
            existing_df = pd.read_csv(output_path, index_col=0)
            print(f"✓ Found {len(existing_df)} already processed documents")
            return set(existing_df.index)
        except Exception as e:
            print(f"Warning: Could not load existing CSV ({e})")
            return set()

    def _save_results(self, results: Dict, output_path: str):
        """
        Thread-safe save of results.

        Args:
            results: Dictionary of results to save
            output_path: Path to save CSV
        """
        with self.write_lock:
            results_df = pd.DataFrame.from_dict(results, orient='index')
            results_df.to_csv(output_path)
