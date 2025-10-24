"""
Main entry point for running summarization experiments.

This script orchestrates running various summarization methods on the dataset,
including baseline methods and the new reflective summarization approach.
"""

import dotenv
from summarizers.baseline import BaselineSummarizer, ThinkingSummarizer
from summarizers.reflective import ReflectiveSummarizer
from experiments.run_experiment import SummarizationExperiment


def main():
    """Run all summarization experiments"""
    dotenv.load_dotenv()

    # Define all summarization methods to test
    summarizers = [
        # Baseline methods
        BaselineSummarizer(enable_thinking=False),
        BaselineSummarizer(enable_thinking=True),

        # Reflective methods with different configurations
        ReflectiveSummarizer(
            surprise_threshold=7.0,
            max_iterations=2,
            top_k_sentences=3
        ),
        ReflectiveSummarizer(
            surprise_threshold=8.0,
            max_iterations=3,
            top_k_sentences=5
        ),
    ]

    # Run experiment
    experiment = SummarizationExperiment(
        summarizers=summarizers,
        output_dir='new_data',
        max_workers=10,
        checkpoint_every=5
    )

    # Run on both base and topic prompts
    input_path = 'data/train_combined_with_prompts.csv'

    for prompt_type in ['base', 'topic']:
        print(f"\n{'#'*60}")
        print(f"# Running with {prompt_type.upper()} prompts")
        print(f"{'#'*60}\n")
        experiment.run(input_path, prompt_type=prompt_type)


if __name__ == "__main__":
    main()
