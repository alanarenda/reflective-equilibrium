# Summary Evaluation Guide

This guide explains how to evaluate generated summaries using ROUGE scores and QA accuracy metrics.

## Overview

The evaluation system provides two main metrics:

1. **ROUGE Scores**: Compare generated summaries against reference summaries using ROUGE-1, ROUGE-2, and ROUGE-L metrics
2. **QA Accuracy**: Test how well the summary captures information by having an LM answer questions using only the summary, then comparing to reference answers

## Code Structure

The evaluation code is organized in the `evaluation/` directory:

```
evaluation/
├── __init__.py           # Package exports
├── evaluator.py          # Main Evaluator class (combines ROUGE + QA)
├── rouge_evaluator.py    # ROUGE-specific evaluation
├── qa_evaluator.py       # QA-specific evaluation
└── metrics.py            # Data classes (RougeScores, QAResult)
```

**Key Components:**
- `Evaluator`: Main class that combines both ROUGE and QA evaluation
- `RougeEvaluator`: Standalone ROUGE scorer (fast, no API needed)
- `QAEvaluator`: Standalone QA evaluator (requires LM API)
- `RougeScores`, `QAResult`: Data classes for structured results

## Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Evaluate a Single Summary File

```bash
python experiments/run_evaluation.py \
    new_data/baseline_thinking_base_summaries.csv \
    --split test \
    --workers 5
```

### 2. Evaluate Without QA (ROUGE Only)

If you want faster evaluation with only ROUGE scores:

```bash
python experiments/run_evaluation.py \
    new_data/baseline_thinking_base_summaries.csv \
    --split test \
    --no-qa
```

### 3. Test on a Small Subset

For testing, evaluate just a few documents:

```bash
python experiments/run_evaluation.py \
    new_data/baseline_thinking_base_summaries.csv \
    --split test \
    --max-docs 10 \
    --workers 2
```

## Command-Line Arguments

```
positional arguments:
  summary_file          Path to CSV file with generated summaries

optional arguments:
  --split {train,val,test}
                        Dataset split to use for reference summaries (default: test)
  --no-qa              Skip QA evaluation (only compute ROUGE)
  --max-docs MAX_DOCS  Maximum number of documents to evaluate (for testing)
  --workers WORKERS    Number of parallel workers (default: 5)
  --model MODEL        Model to use for QA evaluation (default: Qwen/QwQ-32B-Preview)
  --api-key API_KEY    Together API key (or set TOGETHER_API_KEY env var)
```

## Input Format

The summary CSV file should have:
- First column: Document IDs (used as index)
- Column named `summary`: The generated summary text

Example:
```csv
,summary
GAO_GAO-12-604,"This is the generated summary..."
GAO_GAO-13-123,"Another generated summary..."
```

## Output Files

The evaluation creates several output files in the `evaluation_results/` directory:

1. **`{summary_name}_{split}_evaluation.json`**: Full evaluation results including:
   - ROUGE scores for each document
   - QA results with questions, answers, and accuracy scores
   - Success/error status

2. **`{summary_name}_{split}_stats.json`**: Aggregated statistics:
   - Mean ROUGE scores across all documents
   - Mean QA accuracy

3. **`{summary_name}_{split}_rouge_scores.csv`**: ROUGE scores per document in CSV format

4. **`{summary_name}_{split}_qa_results.csv`**: Detailed QA results with all questions and answers

## Evaluation Metrics

### ROUGE Scores

For each document, we compute:
- **ROUGE-1**: Unigram overlap (precision, recall, F-measure)
- **ROUGE-2**: Bigram overlap (precision, recall, F-measure)
- **ROUGE-L**: Longest common subsequence (precision, recall, F-measure)

Higher F-measure scores indicate better summary quality.

### QA Accuracy

For each question in the hierarchical question set:

1. An LM reads only the generated summary (not the original document)
2. The LM answers the question based on the summary
3. Another LM compares the generated answer to the reference answer
4. A similarity score from 0-1 is assigned:
   - 0.0 = Completely different/wrong
   - 0.5 = Partially correct
   - 1.0 = Semantically equivalent

The overall QA accuracy is the mean score across all questions.

## Programmatic Usage

You can also use the evaluator programmatically:

```python
from evaluation import Evaluator
import json

# Initialize evaluator
evaluator = Evaluator(
    api_key="your_api_key",
    model="Qwen/QwQ-32B-Preview"
)

# Evaluate a single summary
with open('data/question_hierarchies_nested.json', 'r') as f:
    hierarchies = json.load(f)

results = evaluator.evaluate_summary(
    doc_id="GAO_GAO-12-604",
    generated_summary="Your generated summary...",
    reference_summary="Reference summary...",
    hierarchy=hierarchies.get("GAO_GAO-12-604"),
    compute_qa=True
)

print(f"ROUGE-1 F-score: {results['rouge_scores']['rouge1_f']:.4f}")
print(f"QA Accuracy: {results['avg_qa_accuracy']:.4f}")
```

## Example Workflow

Here's a complete workflow to evaluate different summarization methods:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set API key
export TOGETHER_API_KEY="your_api_key_here"

# 3. Evaluate baseline (no thinking)
python experiments/run_evaluation.py \
    new_data/baseline_no_thinking_base_summaries.csv \
    --split test \
    --workers 5

# 4. Evaluate baseline (with thinking)
python experiments/run_evaluation.py \
    new_data/baseline_thinking_base_summaries.csv \
    --split test \
    --workers 5

# 5. Evaluate reflective summarizer
python experiments/run_evaluation.py \
    new_data/reflective_t7.0_i2_k3_base_summaries.csv \
    --split test \
    --workers 5

# 6. Compare results
python -c "
import json
import pandas as pd

# Load stats
with open('evaluation_results/baseline_no_thinking_base_summaries_test_stats.json') as f:
    baseline_no = json.load(f)
with open('evaluation_results/baseline_thinking_base_summaries_test_stats.json') as f:
    baseline_yes = json.load(f)
with open('evaluation_results/reflective_t7.0_i2_k3_base_summaries_test_stats.json') as f:
    reflective = json.load(f)

# Create comparison table
comparison = pd.DataFrame([
    {'Method': 'Baseline (No Thinking)', **baseline_no},
    {'Method': 'Baseline (With Thinking)', **baseline_yes},
    {'Method': 'Reflective', **reflective}
])

print(comparison.to_string(index=False))
"
```

## Performance Considerations

### Speed vs. Accuracy

- **ROUGE-only evaluation**: ~1-2 seconds per document
- **ROUGE + QA evaluation**: ~30-60 seconds per document (depends on number of questions)

### Parallel Processing

The evaluation system uses ThreadPoolExecutor for parallel processing:
- Default: 5 workers
- For QA evaluation, use fewer workers (3-5) to avoid API rate limits
- For ROUGE-only, you can use more workers (10-20)

### Checkpointing

Results are automatically checkpointed every 10 documents to prevent data loss.

## Troubleshooting

### API Rate Limits

If you encounter rate limit errors:
1. Reduce the number of workers: `--workers 2`
2. Add delays between requests (modify the evaluator code)
3. Use a different model with higher rate limits

### Out of Memory

For large evaluations:
1. Process in smaller batches using `--max-docs`
2. Reduce the number of workers
3. Disable QA evaluation with `--no-qa`

### Mismatched Document IDs

Ensure your summary CSV uses the same document IDs as the dataset:
- Check that the first column in your CSV matches `doc_id` in the dataset
- Common issue: CSV has integer indices instead of document IDs

## Advanced Usage

### Custom QA Model

Use a different model for QA evaluation:

```bash
python experiments/run_evaluation.py \
    new_data/baseline_thinking_base_summaries.csv \
    --split test \
    --model "meta-llama/Llama-3-70b-chat-hf"
```

### Evaluate on Validation Set

```bash
python experiments/run_evaluation.py \
    new_data/baseline_thinking_base_summaries.csv \
    --split val \
    --workers 5
```

### Using Individual Evaluators

You can use the ROUGE or QA evaluators separately:

```python
from evaluation import RougeEvaluator, QAEvaluator

# ROUGE-only evaluation (fast, no API key needed)
rouge_eval = RougeEvaluator()
rouge_scores = rouge_eval.compute_rouge_scores(
    generated_summary="...",
    reference_summary="..."
)
print(f"ROUGE-1: {rouge_scores.rouge1_fmeasure:.4f}")

# QA-only evaluation
qa_eval = QAEvaluator(api_key="your_key")
qa_results, avg_accuracy = qa_eval.evaluate_document_qa(
    doc_id="doc_123",
    generated_summary="...",
    questions=["Q1", "Q2"],
    reference_answers=["A1", "A2"],
    question_indices=[0, 1],
    paragraph_indices=[0, 0]
)
```

### Custom Evaluation Logic

Extend the `Evaluator` class in [evaluation/evaluator.py](evaluation/evaluator.py) to add custom metrics:

```python
from evaluation import Evaluator

class CustomEvaluator(Evaluator):
    def compute_custom_metric(self, summary: str) -> float:
        # Your custom metric logic
        return score
```

## Citation

If you use this evaluation framework, please cite the GovReport dataset:

```bibtex
@inproceedings{huang2021govreport,
  title={Efficient Attentions for Long Document Summarization},
  author={Huang, Luyang and Cao, Shuyang and Parulian, Nikolaus and Ji, Heng and Wang, Lu},
  booktitle={NAACL},
  year={2021}
}
```
