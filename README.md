# Prompt Injection Evaluation Tool

A comprehensive evaluation tool for prompt injection detection models. This script evaluates model performance on prompt injection datasets and generates detailed analysis reports.

## Features

- Supports multiple model types (Llama Prompt Guard, ProtectAI DeBERTa, PreambleAI, Qualifire, SavantAI, etc.)
- Processes Parquet format datasets automatically
- Generates comprehensive evaluation metrics
- Creates organized output directories with model names and timestamps
- Handles various text column formats automatically
- Provides detailed confusion matrix analysis
- Exports results in multiple formats (JSON, CSV, TXT)

## Prerequisites

### Environment Setup
```bash
conda create -n model-eval python=3.12
conda activate model-eval
python -m pip install torch pandas transformers tqdm pyarrow
```

### Required Dependencies
- torch
- pandas
- transformers
- tqdm
- pyarrow
- numpy

## Project Structure

```
model-eval/
├── model-eval-v1.py          # Main evaluation script
├── models/                   # Model directory
│   ├── Llama-Prompt-Guard-2-86M/
│   └── protectai/
├── datasets/                 # Dataset directory
│   ├── test-00000-of-00001-701d16158af87368.parquet
│   └── train-00000-of-00001-9564e8b05b4757ab.parquet
└── evaluation_results_*/     # Generated output directories
```

## Configuration

### Model Configuration
Edit the model path in `model-eval-v1.py`:

```python
# For Llama Prompt Guard
model_path = "./models/Llama-Prompt-Guard-2-86M/"

# For ProtectAI DeBERTa
model_path = "./models/protectai/"

# For PreambleAI
model_path = "./models/preambleai/"

# For Qualifire
model_path = "./models/qualifire/"

# For SavantAI
model_path = "./models/testsavantai-prompt-injection-defender-large-v0/"
```

### Dataset Configuration
```python
datasets_path = "./datasets/"
```

## Usage

### Basic Usage
```bash
python model-eval-v1.py
```

### Supported Models
- **Llama Prompt Guard 2 (86M)**: Meta's prompt injection detection model
- **ProtectAI DeBERTa v3**: ProtectAI's DeBERTa-based prompt injection detector
- **PreambleAI**: PreambleAI's prompt injection detection model
- **Qualifire**: Qualifire's prompt injection detection model
- **SavantAI**: SavantAI's large prompt injection defender model

## Dataset Format

The script automatically detects and processes:
- **Text columns**: `text`, `prompt`, `input`, `content`, `message`
- **Label columns**: `label`, `target`, `ground_truth`, `gt`
- **Format**: Parquet files with standard structure

## Output Files

The script creates a timestamped directory with the model name: `evaluation_results_{model_name}_{timestamp}/`

### Generated Files

1. **`eval_results_{timestamp}.json`**
   - Complete evaluation results in JSON format
   - Contains all predictions, original labels, and metadata

2. **`eval_summary_{timestamp}.csv`**
   - Flattened results suitable for spreadsheet analysis
   - Includes prediction comparisons (TP/TN/FP/FN)
   - Confidence scores for each prediction

3. **`evaluation_report_{timestamp}.txt`**
   - Human-readable evaluation report
   - Detailed confusion matrix
   - All evaluation metrics with explanations

4. **`evaluation_metrics_{timestamp}.json`**
   - Structured evaluation metrics in JSON format
   - Includes formulas and metadata

## Model Evaluation Results Comparison

Based on evaluation results from August 4, 2025, here's the performance comparison of various models on the same test dataset:

| Model Name | Accuracy | Recall | Precision | F1 Score | FPR |
|------------|----------|--------|-----------|----------|-----|
| **SavantAI Defender Large** | **99.40%** | **98.48%** | **100.00%** | **99.23%** | **0.00%** |
| **Qualifire** | **96.83%** | **92.78%** | **99.19%** | **95.87%** | **0.50%** |
| **ProtectAI v1** | 77.49% | 45.25% | 95.97% | 61.50% | 1.25% |
| **ProtectAI v2** | 76.13% | 41.44% | 96.46% | 57.98% | 1.00% |
| **PreambleAI** | 74.62% | 47.91% | 80.25% | 60.00% | 7.77% |
| **Llama Prompt Guard 2** | 69.18% | 22.81% | 98.36% | 37.04% | 0.25% |

### Detailed Confusion Matrix Comparison

| Model Name | TP | TN | FP | FN | Total Samples |
|------------|----|----|----|----|---------------|
| SavantAI Defender Large | 259 | 399 | 0 | 4 | 662 |
| Qualifire | 244 | 397 | 2 | 19 | 662 |
| ProtectAI v1 | 119 | 394 | 5 | 144 | 662 |
| ProtectAI v2 | 109 | 395 | 4 | 154 | 662 |
| PreambleAI | 126 | 368 | 31 | 137 | 662 |
| Llama Prompt Guard 2 | 60 | 398 | 1 | 203 | 662 |

### Evaluation Metrics

- **Accuracy**: Overall correctness rate
- **Recall (Sensitivity)**: True positive rate
- **Precision**: Positive predictive value
- **False Positive Rate (FPR)**: False alarm rate
- **F1 Score**: Harmonic mean of precision and recall

### Confusion Matrix Elements
- **TP (True Positive)**: Correctly identified injections
- **TN (True Negative)**: Correctly identified benign prompts
- **FP (False Positive)**: Benign prompts classified as injections
- **FN (False Negative)**: Missed injection attempts

## Example Output Structure

```
evaluation_results_Llama-Prompt-Guard-2-86M_20250804_143022/
├── eval_results_20250804_143022.json
├── eval_summary_20250804_143022.csv
├── evaluation_report_20250804_143022.txt
└── evaluation_metrics_20250804_143022.json
```

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Ensure model files are properly downloaded
   - Check model path configuration
   - Verify safetensors format compatibility

2. **Dataset Format Issues**
   - Verify Parquet file integrity
   - Check column naming conventions
   - Ensure proper text encoding

3. **Memory Issues**
   - Consider processing datasets in batches
   - Monitor GPU/CPU memory usage
   - Reduce max_length parameter if needed

## Model Sources

- [Llama Prompt Guard 86M](https://huggingface.co/meta-llama/Prompt-Guard-86M)
- [ProtectAI DeBERTa v3](https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2)

## License

This evaluation tool is for research and evaluation purposes. Please check individual model licenses for usage restrictions.