# Prompt Injection Evaluation Tool

A comprehensive evaluation tool for prompt injection detection models. This script evaluates model performance on prompt injection datasets and generates detailed analysis reports.

## Features

- Supports multiple model types (Llama Prompt Guard, ProtectAI DeBERTa, etc.)
- Processes Parquet format datasets automatically
- Generates comprehensive evaluation metrics
- Creates organized output directories with model names and timestamps
- Handles various text column formats automatically
- Provides detailed confusion matrix analysis
- Exports results in multiple formats (JSON, CSV, TXT)

## Prerequisites

### Environment Setup
```bash
conda create -n prompt-guard python=3.12
conda activate prompt-guard
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
prompt-injection-eval/
├── model-eval-v3.py          # Main evaluation script
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
Edit the model path in `model-eval-v3.py`:

```python
# For Llama Prompt Guard
model_path = "./models/Llama-Prompt-Guard-2-86M/"

# For ProtectAI DeBERTa
model_path = "./models/protectai/"
```

### Dataset Configuration
```python
datasets_path = "./datasets/"
```

## Usage

### Basic Usage
```bash
python model-eval-v3.py
```

### Supported Models
- **Llama Prompt Guard 2 (86M)**: Meta's prompt injection detection model
- **ProtectAI DeBERTa v3**: ProtectAI's DeBERTa-based prompt injection detector

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

## Advanced Features

### Automatic Column Detection
The script intelligently identifies:
- Text columns containing prompts to analyze
- Label columns for ground truth comparison
- Additional metadata columns for context

### Error Handling
- Graceful handling of malformed inputs
- Detailed error logging
- Partial results preservation

### Flexible Input Format
- Supports various dataset structures
- Automatic text preprocessing
- Configurable text truncation (512 tokens max)

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