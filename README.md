# Prompt Injection Evaluation Tool

A comprehensive evaluation tool for prompt injection detection models. This tool includes dataset merging capabilities and supports evaluation of multiple model types with unified CSV datasets.

## Features

- **Dataset Management**: Merge multiple format datasets (CSV, TSV, JSON, Parquet) into unified format, defaults to 20265 samples from various sources.
- **Short Dataset Creation**: Create balanced smaller datasets for faster testing and development
- **Model Support**: Multiple model types (Llama Prompt Guard, ProtectAI DeBERTa, PreambleAI, Qualifire, SavantAI, etc.)
- **Binary Classification**: Standardized binary labels (0=Safe, 1=Threat) with original label preservation
- **Comprehensive Metrics**: Detailed evaluation metrics including confusion matrix analysis
- **Organized Output**: Results saved in structured directories under `results/`
- **Command Line Interface**: Flexible CLI for evaluation with sampling support

## Prerequisites

### Environment Setup
```bash
conda create -n model-eval python=3.12
conda activate model-eval
python -m pip install torch pandas transformers tqdm pyarrow -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## Project Structure

```
prompt-injection-eval/
├── merge_datasets.py         # Dataset merging script
├── model-eval-v2.0.py       # Main evaluation script v2.0
├── model-eval-v1.1.py       # Legacy evaluation script
├── datasets/                # Dataset directory
│   ├── create_short_dataset.py  # Short dataset creation script
│   ├── dataset-source/      # Original datasets in various formats
│   │   ├── *.csv            # CSV datasets
│   │   ├── *.tsv            # TSV datasets
│   │   ├── *.json           # JSON datasets
│   │   └── *.parquet        # Parquet datasets
│   ├── merged_datasets.csv  # Unified merged dataset
│   └── merged_datasets_short_*.csv  # Balanced short datasets
├── models/                  # Model directory
│   ├── deepset-deberta/
│   ├── preambleai/
│   ├── qualifire/
│   └── protectaiv1/
│   └── protectaiv2/
└── results/                 # Evaluation results
    └── evaluation_results_*/
```

## Quick Start

### Step 1: Merge Datasets (Optional)
First, merge all datasets into a unified CSV format:

```bash
cd datasets
python merge_datasets.py
```

This will:
- Scan `datasets/dataset-source/` for all supported formats
- Standardize column names and labels
- Convert labels to binary format (0=Safe, 1=Threat)
- Save unified dataset to `datasets/merged_datasets.csv`

### Step 2: Evaluate Models
Run model evaluation using the merged dataset:

```bash
# Basic evaluation
# Default dataset: datasets/merged_datasets.csv, contains 20265 samples
# Default model: ./models/deepset-deberta/
python model-eval-v2.0.py --model ./models/deepset-deberta/

# With specific dataset (using short dataset for faster evaluation)
python model-eval-v2.0.py --model ./models/deepset-deberta/ --dataset datasets/merged_datasets_short_1000.csv

# Sample evaluation (1000 samples)
python model-eval-v2.0.py --model ./models/deepset-deberta/ --sample 1000
```

## Model Evaluation Results Comparison

### Evaluation Results on Full Dataset (20,265 samples)

| Model Name | Accuracy | Recall | Precision | FPR | F1 Score |
|------------|----------|--------|-----------|-----|----------|
| **Qualifire** | **0.9457** | **0.9097** | **0.9543** | **0.0297** | **0.9315** |
| **Vijil mBERT** | **0.9338** | **0.8858** | **0.9477** | **0.0334** | **0.9157** |
| **PreambleAI** | **0.9259** | **0.8469** | **0.9664** | **0.0201** | **0.9027** |
| **SavantAI** | **0.8646** | **0.8994** | **0.7942** | **0.1591** | **0.8435** |
| **ProtectAI v2** | **0.8453** | **0.7238** | **0.8732** | **0.0717** | **0.7915** |
| **Llama Prompt Guard** | **0.7906** | **0.5552** | **0.8860** | **0.0487** | **0.6826** |
| **ProtectAI v1** | **0.7076** | **0.3023** | **0.9293** | **0.0157** | **0.4562** |
| **Deepset DeBERTa** | **0.5557** | **0.9856** | **0.4770** | **0.7377** | **0.6429** |

### Evaluation Results on 3,000 Sample Dataset

| Model Name | Accuracy | Recall | Precision | FPR | F1 Score |
|------------|----------|--------|-----------|-----|----------|
| **Qualifire** | **0.8840** | **0.8330** | **0.9162** | **0.0695** | **0.8726** |
| **SavantAI** | **0.8277** | **0.8456** | **0.8035** | **0.1887** | **0.8240** |
| **PreambleAI** | **0.8210** | **0.7086** | **0.8942** | **0.0765** | **0.7906** |
| **Vijil mBERT** | **0.8160** | **0.7086** | **0.8825** | **0.0860** | **0.7860** |
| **ProtectAI v2** | **0.8123** | **0.7121** | **0.8709** | **0.0962** | **0.7835** |
| **Llama Prompt Guard** | **0.7050** | **0.4724** | **0.8387** | **0.0829** | **0.6044** |
| **Deepset DeBERTa** | **0.6380** | **0.9504** | **0.5726** | **0.6469** | **0.7147** |
| **ProtectAI v1** | **0.6380** | **0.3033** | **0.8298** | **0.0567** | **0.4442** |

### Evaluation Results on 2,000 Sample Dataset

| Model Name | Accuracy | Recall | Precision | FPR | F1 Score |
|------------|----------|--------|-----------|-----|----------|
| **Qualifire** | **0.8465** | **0.7867** | **0.8917** | **0.0944** | **0.8359** |
| **ProtectAI v2** | **0.8070** | **0.7254** | **0.8645** | **0.1123** | **0.7888** |
| **SavantAI** | **0.8040** | **0.8119** | **0.7974** | **0.2038** | **0.8046** |
| **PreambleAI** | **0.7880** | **0.6791** | **0.8654** | **0.1044** | **0.7610** |
| **Vijil mBERT** | **0.7795** | **0.6640** | **0.8605** | **0.1064** | **0.7496** |
| **Llama Prompt Guard** | **0.6630** | **0.4286** | **0.8008** | **0.1054** | **0.5583** |
| **Deepset DeBERTa** | **0.6420** | **0.9306** | **0.5884** | **0.6431** | **0.7210** |
| **ProtectAI v1** | **0.6095** | **0.2968** | **0.7825** | **0.0815** | **0.4303** |

### Evaluation Results on 1,000 Sample Dataset

| Model Name | Accuracy | Recall | Precision | FPR | F1 Score |
|------------|----------|--------|-----------|-----|----------|
| **Qualifire** | **0.8140** | **0.7856** | **0.8229** | **0.1592** | **0.8038** |
| **SavantAI** | **0.7730** | **0.8144** | **0.7425** | **0.2660** | **0.7768** |
| **ProtectAI v2** | **0.7660** | **0.6948** | **0.7967** | **0.1670** | **0.7423** |
| **PreambleAI** | **0.7480** | **0.6619** | **0.7848** | **0.1709** | **0.7181** |
| **Vijil mBERT** | **0.7410** | **0.6515** | **0.7783** | **0.1748** | **0.7093** |
| **Deepset DeBERTa** | **0.6430** | **0.9320** | **0.5825** | **0.6291** | **0.7169** |
| **Llama Prompt Guard** | **0.6360** | **0.4330** | **0.7023** | **0.1728** | **0.5357** |
| **ProtectAI v1** | **0.5970** | **0.3320** | **0.6708** | **0.1534** | **0.4441** |



### Key Findings

**Top Performers (Full Dataset - 20,265 samples):**
1. **Qualifire** - Best overall performance with 94.57% accuracy and excellent balance of metrics
2. **Vijil mBERT** - Strong second with 93.38% accuracy and high precision
3. **PreambleAI** - High precision (96.64%) with good overall performance

**Consistency Across Dataset Sizes:**
- **Qualifire** consistently ranks #1 across all dataset sizes
- **SavantAI** show strong performance on smaller datasets
- **ProtectAI v1** consistently shows low recall but high precision
- **Deepset DeBERTa** has very high recall but poor precision (high false positive rate)

**Trade-offs:**
- **High Precision, Lower Recall**: PreambleAI, Llama Prompt Guard, ProtectAI v1
- **High Recall, Lower Precision**: Deepset DeBERTa
- **Balanced Performance**: Qualifire, Vijil mBERT, SavantAI

## Dataset Management

### Supported Input Formats
- **CSV**: Comma-separated values
- **TSV**: Tab-separated values  
- **JSON**: JavaScript Object Notation
- **Parquet**: Apache Parquet format

### Available Datasets

The project includes one full merged dataset and three pre-generated short datasets with balanced distribution:

#### Dataset Overview

| Dataset | Target Samples | Actual Samples | Safe (0) | Threat (1) |
|---------|----------------|----------------|----------|------------|
| merged_datasets.csv | Full | 20,496 | 12,076 (58.9%) | 8,420 (41.1%) |
| merged_datasets_short_1000.csv | 1,036 | 1,000 | 515 (51.5%) | 485 (48.5%) |
| merged_datasets_short_2000.csv | 2,340 | 2,000 | 1,006 (50.3%) | 994 (49.7%) |
| merged_datasets_short_3000.csv | 3,840 | 3,000 | 1,569 (52.3%) | 1,431 (47.7%) |

#### Source Distribution

| Source Dataset | merged_datasets.csv | 1000.csv | 2000.csv | 3000.csv |
|----------------|---------------------|----------|----------|----------|
| **Total Samples** | **20,496** | **1,000** | **2,000** | **3,000** |
| qualifire_prompt-injections-benchmark_test | 5,000 (24.4%) | 116 (11.6%) | 260 (13.0%) | 427 (14.2%) |
| train-00000-of-00001 | 8,236 (40.2%) | 115 (11.5%) | 260 (13.0%) | 427 (14.2%) |
| test-00000-of-00001 | 2,060 (10.1%) | 115 (11.5%) | 260 (13.0%) | 426 (14.2%) |
| allenai_wildjailbreak-eval | 2,210 (10.8%) | 115 (11.5%) | 260 (13.0%) | 427 (14.2%) |
| jackhhao_jailbreak-classification-jailbreak_dataset_full | 1,998 (9.7%) | 115 (11.5%) | 260 (13.0%) | 427 (14.2%) |
| train-00000-of-00001-9564e8b05b4757ab | 546 (2.7%) | 115 (11.5%) | 260 (13.0%) | 426 (14.2%) |
| PurpleLlama_CybersecurityBenchmarks_prompt_injection | 251 (1.2%) | 115 (11.5%) | 245 (12.2%) | 245 (8.2%) |
| test-00000-of-00001-701d16158af87368 | 116 (0.6%) | 115 (11.5%) | 116 (5.8%) | 116 (3.9%) |
| rubend18_ChatGPT-Jailbreak-Prompts | 79 (0.4%) | 79 (7.9%) | 79 (4.0%) | 79 (2.6%) |

*merged_datasets.csv actually has 20265 samples, 231 samples are duplicates and are removed.*

#### Label Distribution by Source (merged_datasets.csv)

| Source Dataset | Total | Safe (0) | Threat (1) | Safe % | Threat % |
|----------------|-------|----------|------------|--------|----------|
| qualifire_prompt-injections-benchmark_test | 5,000 | 3,001 | 1,999 | 60.0% | 40.0% |
| train-00000-of-00001 | 8,236 | 5,740 | 2,496 | 69.7% | 30.3% |
| test-00000-of-00001 | 2,060 | 1,410 | 650 | 68.4% | 31.6% |
| allenai_wildjailbreak-eval | 2,210 | 210 | 2,000 | 9.5% | 90.5% |
| jackhhao_jailbreak-classification-jailbreak_dataset_full | 1,998 | 1,332 | 666 | 66.7% | 33.3% |
| train-00000-of-00001-9564e8b05b4757ab | 546 | 343 | 203 | 62.8% | 37.2% |
| PurpleLlama_CybersecurityBenchmarks_prompt_injection | 251 | 0 | 251 | 0.0% | 100.0% |
| test-00000-of-00001-701d16158af87368 | 116 | 56 | 60 | 48.3% | 51.7% |
| rubend18_ChatGPT-Jailbreak-Prompts | 79 | 74 | 5 | 93.7% | 6.3% |

### Short Dataset Creation

The `create_short_dataset.py` script allows you to create balanced smaller datasets from the full merged dataset for faster testing and development:

#### Features
- **Balanced Sampling**: Maintains proportional representation from each source dataset
- **Label Balance**: Preserves label distribution within each source when possible
- **Configurable Size**: Specify target number of samples
- **Reproducible**: Uses random seed for consistent results
- **Auto-naming**: Generates descriptive filenames based on actual sample count

#### Usage

```bash
cd datasets

# Basic usage - creates 1000 samples by default
python create_short_dataset.py

# Custom sample size

python create_short_dataset.py --samples=1036
#python create_short_dataset.py --samples=2340
#python create_short_dataset.py --samples=3840


# Specify input and output files
python create_short_dataset.py --input merged_datasets.csv --output my_dataset.csv --samples 500

# With custom random seed
python create_short_dataset.py --samples 1000 --seed 123
```

#### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--input` | Input CSV file path | `merged_datasets.csv` |
| `--output` | Output CSV file path (auto-generated if not specified) | Auto-generated |
| `--samples` | Target number of samples to extract | 1000 |
| `--seed` | Random seed for reproducibility | 42 |

#### Example Output

```
Creating balanced short dataset with target of 1000 samples...
Loading dataset from: merged_datasets.csv
Loaded dataset with 20265 rows

=== Original Dataset Distribution ===
Label distribution:
  0 (Safe): 11,847 (58.4%)
  1 (Threat): 8,418 (41.6%)

=== Sampling Strategy ===
Target samples: 1000
Total sources: 9
Base samples per source: 111
Remainder to distribute: 1

Final short dataset saved to: merged_datasets_short_1000.csv
```

### Dataset Structure

All datasets follow the same structure after merging:

| Column | Description | Type |
|--------|-------------|------|
| text | The prompt/text content | string |
| label | Binary label (0=Safe, 1=Threat) | integer |
| source | Source dataset identifier | string |
| original_label | Original label from source | string |

### Label Mapping

The merging process standardizes all labels to binary format:
- **0 (Safe)**: Normal, benign prompts
- **1 (Threat)**: Prompt injection attempts, jailbreaks, malicious prompts

## Model Configuration

### Supported Models
- **Llama Prompt Guard 2 (86M)**: `./models/Llama-Prompt-Guard-2-86M/`
- **ProtectAI DeBERTa v3**: `./models/protectaiv2/`
- **ProtectAI DeBERTa v1**: `./models/protectaiv1/`
- **PreambleAI**: `./models/preambleai/`
- **Qualifire**: `./models/qualifire/`
- **SavantAI**: `./models/testsavantai-prompt-injection-defender-large-v0/`
- **Deepset DeBERTa**: `./models/deepset-deberta/`
- **Vijil mBERT**: `./models/vijil-mbert-prompt-injection/`
- **YOUR OWN MODEL ...**

### Model Downloads
```bash
# Example model downloads using huggingface-cli
huggingface-cli download PreambleAI/prompt-injection-defense --local-dir ./models/preambleai/
huggingface-cli download protectai/deberta-v3-base-prompt-injection-v2 --local-dir ./models/protectaiv2/
```

## Output Files

Results are saved in timestamped directories under `results/`:
```
results/evaluation_results_{model_name}_{timestamp}/
├── eval_results_{timestamp}.json      # Detailed results with predictions
├── eval_summary_{timestamp}.csv       # Flattened results for analysis
├── evaluation_report_{timestamp}.txt  # Human-readable report
└── evaluation_metrics_{timestamp}.json # Structured metrics
```

## Evaluation Metrics

The tool provides comprehensive evaluation metrics:
- **Accuracy**: Overall classification accuracy
- **Precision**: Threat detection precision
- **Recall**: Threat detection recall (sensitivity)
- **F1 Score**: Harmonic mean of precision and recall
- **False Positive Rate (FPR)**: Rate of false threat detection
- **Confusion Matrix**: Detailed classification breakdown

## Advanced Usage

### Custom Model Integration
To add your own model, create a directory under `models/` and ensure it follows the HuggingFace transformers format.

### Batch Evaluation
```bash
# Evaluate multiple models with the same dataset
 for model in deepset-deberta testsavantai-prompt-injection-defender-large-v0 Llama-Prompt-Guard-2-86M protectaiv1 protectaiv2 preambleai qualifire; do
    python model-eval-v2.0.py --model ./models/$model/ --dataset datasets/merged_datasets_short_2000.csv
done
```

### Sampling Strategies
```bash
# Random sampling from full dataset
python model-eval-v2.0.py --model ./models/deepset-deberta/ --sample 500

# Use pre-created balanced short dataset
python model-eval-v2.0.py --model ./models/deepset-deberta/ --dataset datasets/merged_datasets_short_1000.csv
```

## Version History

- **v2.0**: Enhanced evaluation pipeline with better output formatting
- **v1.1**: Added sampling support and improved metrics
- **v1.0**: Initial release with basic model evaluation

## Model Sources

### 3.1 testsavantai/prompt-injection-defender-large-v0
https://huggingface.co/testsavantai/prompt-injection-defender-large-v0

### 3.2 Qualifire
https://huggingface.co/qualifire/prompt-injection-sentinel

### 3.3 ProtectAI deberta-v3-base-prompt-injection-v2
https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2

### 3.4 Meta Llama Prompt-Guard-86M
https://huggingface.co/meta-llama/Prompt-Guard-86M/tree/main

### 3.5 vijil/mbert-prompt-injection
https://huggingface.co/vijil/mbert-prompt-injection

### 3.6 PreambleAI/prompt-injection-defense
https://huggingface.co/PreambleAI/prompt-injection-defense/tree/main
*Training dataset not specified*

## Dataset Sources

### 4.1 Deepset
Qualifire used this dataset for training:
https://huggingface.co/datasets/deepset/prompt-injections

Clone of this dataset used by testsavantai/prompt-injection-defender-large-v0:
https://huggingface.co/datasets/JasperLS/prompt-injections

### 4.2 allenai/wildjailbreak
Qualifire used this dataset for training:
https://huggingface.co/datasets/allenai/wildjailbreak

### 4.3 jackhhao/jailbreak-classification
ProtectAI and Qualifire used this dataset for training:
https://huggingface.co/datasets/jackhhao/jailbreak-classification

### 4.4 qualifire/prompt-injections-benchmark
https://huggingface.co/datasets/qualifire/prompt-injections-benchmark

### 4.5 xTRam1/safe-guard-prompt-injection
https://huggingface.co/datasets/xTRam1/safe-guard-prompt-injection

### 4.6 rubend18/ChatGPT-Jailbreak-Prompts
ProtectAI and testsavantai/prompt-injection-defender-large-v0 used this dataset for training:
https://huggingface.co/datasets/rubend18/ChatGPT-Jailbreak-Prompts

### 4.7 NVIDIA nvidia/Aegis-AI-Content-Safety-Dataset-2.0
Prisma AIRS used this dataset for training:
https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0

### 4.8 Purple Llama CyberSecEval
Llama Prompt Guard used this dataset for training:
https://github.com/meta-llama/PurpleLlama/tree/23156b70efb596831c02c6461fc42da1f75988ec/CybersecurityBenchmarks

## License

This project is open source. Please check individual model licenses for their respective usage terms.