"""
Example script showing how to use the evaluation system.

This demonstrates:
1. Loading data
2. Running evaluation on a single document
3. Comparing multiple summarization methods
"""

import os
import json
import pandas as pd
from evaluation import Evaluator


def example_single_document_evaluation():
    """Example: Evaluate a single document's summary."""
    print("="*60)
    print("EXAMPLE 1: Single Document Evaluation")
    print("="*60)

    # Get API key
    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        print("ERROR: Set TOGETHER_API_KEY environment variable")
        return

    # Initialize evaluator
    evaluator = Evaluator(api_key=api_key)

    # Load a sample document
    test_data = pd.read_csv("data/test_combined.csv")
    sample_doc = test_data.iloc[0]

    doc_id = sample_doc['doc_id']
    reference_summary = sample_doc['summary']

    # Load hierarchies
    with open('data/question_hierarchies_nested.json', 'r') as f:
        hierarchies = json.load(f)

    # Example generated summary (you would normally load this from your results)
    generated_summary = reference_summary  # Using reference as example

    # Evaluate
    print(f"\nEvaluating document: {doc_id}")
    print("Computing ROUGE scores and QA accuracy...")

    results = evaluator.evaluate_summary(
        doc_id=doc_id,
        generated_summary=generated_summary,
        reference_summary=reference_summary,
        hierarchy=hierarchies.get(doc_id),
        compute_qa=True  # Set to False for faster ROUGE-only evaluation
    )

    # Print results
    print("\nROUGE Scores:")
    rouge = results['rouge_scores']
    print(f"  ROUGE-1: P={rouge['rouge1_p']:.4f}, R={rouge['rouge1_r']:.4f}, F={rouge['rouge1_f']:.4f}")
    print(f"  ROUGE-2: P={rouge['rouge2_p']:.4f}, R={rouge['rouge2_r']:.4f}, F={rouge['rouge2_f']:.4f}")
    print(f"  ROUGE-L: P={rouge['rougeL_p']:.4f}, R={rouge['rougeL_r']:.4f}, F={rouge['rougeL_f']:.4f}")

    if results['avg_qa_accuracy'] is not None:
        print(f"\nQA Accuracy: {results['avg_qa_accuracy']:.4f}")
        print(f"Number of questions: {len(results['qa_results'])}")

        # Show first few QA results
        print("\nSample QA Results:")
        for i, qa in enumerate(results['qa_results'][:3]):
            print(f"\n  Q{i+1}: {qa['question'][:100]}...")
            print(f"  Accuracy: {qa['accuracy']:.2f}")

    print("\n" + "="*60 + "\n")


def example_batch_evaluation():
    """Example: Evaluate multiple summaries from a CSV file."""
    print("="*60)
    print("EXAMPLE 2: Batch Evaluation")
    print("="*60)

    # Check if summary files exist
    summary_files = [
        "new_data/baseline_no_thinking_base_summaries.csv",
        "new_data/baseline_thinking_base_summaries.csv",
    ]

    available_files = [f for f in summary_files if os.path.exists(f)]

    if not available_files:
        print("No summary files found in new_data/")
        print("Generate summaries first using summarize.py")
        return

    print(f"\nFound {len(available_files)} summary files to evaluate")
    print("\nTo evaluate them, run:")

    for file in available_files:
        print(f"\n  python experiments/run_evaluation.py \\")
        print(f"      {file} \\")
        print(f"      --split test \\")
        print(f"      --max-docs 10 \\  # Limit for testing")
        print(f"      --workers 3")

    print("\n" + "="*60 + "\n")


def example_rouge_only():
    """Example: Fast ROUGE-only evaluation (no QA)."""
    print("="*60)
    print("EXAMPLE 3: ROUGE-Only Evaluation (Fast)")
    print("="*60)

    # Get API key (not needed for ROUGE-only, but Evaluator requires it)
    api_key = os.environ.get("TOGETHER_API_KEY", "dummy")

    # Initialize evaluator
    evaluator = Evaluator(api_key=api_key)

    # Load sample data
    test_data = pd.read_csv("data/test_combined.csv", nrows=5)

    print(f"\nComputing ROUGE scores for {len(test_data)} documents...")

    results = []
    for idx, row in test_data.iterrows():
        # Using reference summary as both reference and generated (example only)
        rouge_scores = evaluator.compute_rouge_scores(
            generated_summary=row['summary'],
            reference_summary=row['summary']
        )

        results.append({
            'doc_id': row['doc_id'],
            **rouge_scores.to_dict()
        })

    # Create DataFrame
    results_df = pd.DataFrame(results)

    print("\nROUGE Scores Summary:")
    print(results_df[['doc_id', 'rouge1_f', 'rouge2_f', 'rougeL_f']].to_string(index=False))

    print(f"\nMean ROUGE-1 F-score: {results_df['rouge1_f'].mean():.4f}")
    print(f"Mean ROUGE-2 F-score: {results_df['rouge2_f'].mean():.4f}")
    print(f"Mean ROUGE-L F-score: {results_df['rougeL_f'].mean():.4f}")

    print("\n" + "="*60 + "\n")


def example_compare_methods():
    """Example: Compare evaluation results from different methods."""
    print("="*60)
    print("EXAMPLE 4: Compare Evaluation Results")
    print("="*60)

    # Check for evaluation result files
    result_dir = "evaluation_results"

    if not os.path.exists(result_dir):
        print(f"\nNo results found in {result_dir}/")
        print("Run evaluations first using experiments/run_evaluation.py")
        return

    # Look for stats files
    stats_files = [
        f for f in os.listdir(result_dir)
        if f.endswith('_stats.json')
    ]

    if not stats_files:
        print("No evaluation stats files found")
        return

    print(f"\nFound {len(stats_files)} evaluation results:")

    # Load and compare
    comparison_data = []
    for stats_file in stats_files:
        with open(os.path.join(result_dir, stats_file), 'r') as f:
            stats = json.load(f)

        # Extract method name from filename
        method_name = stats_file.replace('_test_stats.json', '').replace('_val_stats.json', '')

        comparison_data.append({
            'Method': method_name,
            'ROUGE-1': stats.get('rouge1_f_mean', 0),
            'ROUGE-2': stats.get('rouge2_f_mean', 0),
            'ROUGE-L': stats.get('rougeL_f_mean', 0),
            'QA Accuracy': stats.get('qa_accuracy_mean', 0),
            'Num Docs': stats.get('num_documents', 0)
        })

    # Create comparison table
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('ROUGE-L', ascending=False)

    print("\nComparison of Methods:")
    print(comparison_df.to_string(index=False))

    # Find best method
    best_rouge = comparison_df.iloc[0]
    print(f"\nBest method by ROUGE-L: {best_rouge['Method']}")

    if comparison_df['QA Accuracy'].max() > 0:
        best_qa = comparison_df.loc[comparison_df['QA Accuracy'].idxmax()]
        print(f"Best method by QA Accuracy: {best_qa['Method']}")

    print("\n" + "="*60 + "\n")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("EVALUATION EXAMPLES")
    print("="*60 + "\n")

    # Example 1: Single document evaluation
    # Uncomment to run (requires API key and can be slow)
    # example_single_document_evaluation()

    # Example 2: Batch evaluation instructions
    example_batch_evaluation()

    # Example 3: Fast ROUGE-only evaluation
    example_rouge_only()

    # Example 4: Compare results
    example_compare_methods()

    print("\nTo enable Example 1 (single document with QA):")
    print("  1. Uncomment the line in main()")
    print("  2. Set TOGETHER_API_KEY environment variable")
    print("  3. Run: python example_evaluation.py")


if __name__ == "__main__":
    main()
