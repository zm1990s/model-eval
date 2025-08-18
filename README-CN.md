# 提示注入检测模型评估工具

针对提示注入检测模型的综合评估工具。该工具包含数据集合并功能，支持多种模型类型的评估，并使用统一的CSV数据集格式。

## 功能特性

- **数据集管理**：将多种格式的数据集（CSV、TSV、JSON、Parquet）合并为统一格式，默认包含来自各种来源的20265个样本
- **短数据集创建**：创建平衡的较小数据集，用于更快的测试和开发
- **模型支持**：支持多种模型类型（Llama Prompt Guard、ProtectAI DeBERTa、PreambleAI、Qualifire、SavantAI等）
- **二元分类**：标准化的二元标签（0=安全，1=威胁），保留原始标签
- **全面指标**：详细的评估指标，包括混淆矩阵分析
- **有序输出**：结果保存在`results/`下的结构化目录中
- **命令行界面**：灵活的CLI，支持采样评估

## 环境要求

### 环境设置
```bash
conda create -n model-eval python=3.12
conda activate model-eval
python -m pip install torch pandas transformers tqdm pyarrow -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 项目结构

```
prompt-injection-eval/
├── merge_datasets.py         # 数据集合并脚本
├── model-eval-v2.0.py       # 主评估脚本 v2.0
├── model-eval-v1.1.py       # 传统评估脚本
├── datasets/                # 数据集目录
│   ├── create_short_dataset.py  # 短数据集创建脚本
│   ├── dataset-source/      # 各种格式的原始数据集
│   │   ├── *.csv            # CSV数据集
│   │   ├── *.tsv            # TSV数据集
│   │   ├── *.json           # JSON数据集
│   │   └── *.parquet        # Parquet数据集
│   ├── merged_datasets.csv  # 统一合并数据集
│   └── merged_datasets_short_*.csv  # 平衡的短数据集
├── models/                  # 模型目录
│   ├── deepset-deberta/
│   ├── preambleai/
│   ├── qualifire/
│   └── protectaiv1/
│   └── protectaiv2/
└── results/                 # 评估结果
    └── evaluation_results_*/
```

## 快速开始

### 步骤1：合并数据集（可选）
首先，将所有数据集合并为统一的CSV格式：

```bash
cd datasets
python merge_datasets.py
```

这将：
- 扫描`datasets/dataset-source/`下的所有支持格式
- 标准化列名和标签
- 将标签转换为二元格式（0=安全，1=威胁）
- 将统一数据集保存到`datasets/merged_datasets.csv`

### 步骤2：评估模型
使用合并的数据集运行模型评估：

```bash
# 基本评估
# 默认数据集：datasets/merged_datasets.csv，包含20265个样本
# 默认模型：./models/deepset-deberta/
python model-eval-v2.0.py --model ./models/deepset-deberta/

# 使用特定数据集（使用短数据集进行更快评估）
python model-eval-v2.0.py --model ./models/deepset-deberta/ --dataset datasets/merged_datasets_short_1000.csv

# 采样评估（1000个样本）
python model-eval-v2.0.py --model ./models/deepset-deberta/ --sample 1000
```

## 模型评估结果对比

### 完整数据集评估结果（20,265个样本）

| 模型名称 | 准确率 | 召回率 | 精确率 | FPR | F1分数 |
|----------|--------|--------|--------|-----|--------|
| **Qualifire** | **0.9457** | **0.9097** | **0.9543** | **0.0297** | **0.9315** |
| **Vijil mBERT** | **0.9338** | **0.8858** | **0.9477** | **0.0334** | **0.9157** |
| **PreambleAI** | **0.9259** | **0.8469** | **0.9664** | **0.0201** | **0.9027** |
| **SavantAI** | **0.8646** | **0.8994** | **0.7942** | **0.1591** | **0.8435** |
| **ProtectAI v2** | **0.8453** | **0.7238** | **0.8732** | **0.0717** | **0.7915** |
| **Llama Prompt Guard** | **0.7906** | **0.5552** | **0.8860** | **0.0487** | **0.6826** |
| **ProtectAI v1** | **0.7076** | **0.3023** | **0.9293** | **0.0157** | **0.4562** |
| **Deepset DeBERTa** | **0.5557** | **0.9856** | **0.4770** | **0.7377** | **0.6429** |

### 3000样本数据集评估结果

| 模型名称 | 准确率 | 召回率 | 精确率 | FPR | F1分数 |
|----------|--------|--------|--------|-----|--------|
| **Qualifire** | **0.8840** | **0.8330** | **0.9162** | **0.0695** | **0.8726** |
| **SavantAI** | **0.8277** | **0.8456** | **0.8035** | **0.1887** | **0.8240** |
| **PreambleAI** | **0.8210** | **0.7086** | **0.8942** | **0.0765** | **0.7906** |
| **Vijil mBERT** | **0.8160** | **0.7086** | **0.8825** | **0.0860** | **0.7860** |
| **ProtectAI v2** | **0.8123** | **0.7121** | **0.8709** | **0.0962** | **0.7835** |
| **Llama Prompt Guard** | **0.7050** | **0.4724** | **0.8387** | **0.0829** | **0.6044** |
| **Deepset DeBERTa** | **0.6380** | **0.9504** | **0.5726** | **0.6469** | **0.7147** |
| **ProtectAI v1** | **0.6380** | **0.3033** | **0.8298** | **0.0567** | **0.4442** |

### 2000样本数据集评估结果

| 模型名称 | 准确率 | 召回率 | 精确率 | FPR | F1分数 |
|----------|--------|--------|--------|-----|--------|
| **Qualifire** | **0.8465** | **0.7867** | **0.8917** | **0.0944** | **0.8359** |
| **ProtectAI v2** | **0.8070** | **0.7254** | **0.8645** | **0.1123** | **0.7888** |
| **SavantAI** | **0.8040** | **0.8119** | **0.7974** | **0.2038** | **0.8046** |
| **PreambleAI** | **0.7880** | **0.6791** | **0.8654** | **0.1044** | **0.7610** |
| **Vijil mBERT** | **0.7795** | **0.6640** | **0.8605** | **0.1064** | **0.7496** |
| **Llama Prompt Guard** | **0.6630** | **0.4286** | **0.8008** | **0.1054** | **0.5583** |
| **Deepset DeBERTa** | **0.6420** | **0.9306** | **0.5884** | **0.6431** | **0.7210** |
| **ProtectAI v1** | **0.6095** | **0.2968** | **0.7825** | **0.0815** | **0.4303** |

### 1000样本数据集评估结果

| 模型名称 | 准确率 | 召回率 | 精确率 | FPR | F1分数 |
|----------|--------|--------|--------|-----|--------|
| **Qualifire** | **0.8140** | **0.7856** | **0.8229** | **0.1592** | **0.8038** |
| **SavantAI** | **0.7730** | **0.8144** | **0.7425** | **0.2660** | **0.7768** |
| **ProtectAI v2** | **0.7660** | **0.6948** | **0.7967** | **0.1670** | **0.7423** |
| **PreambleAI** | **0.7480** | **0.6619** | **0.7848** | **0.1709** | **0.7181** |
| **Vijil mBERT** | **0.7410** | **0.6515** | **0.7783** | **0.1748** | **0.7093** |
| **Deepset DeBERTa** | **0.6430** | **0.9320** | **0.5825** | **0.6291** | **0.7169** |
| **Llama Prompt Guard** | **0.6360** | **0.4330** | **0.7023** | **0.1728** | **0.5357** |
| **ProtectAI v1** | **0.5970** | **0.3320** | **0.6708** | **0.1534** | **0.4441** |



### 关键发现

**顶级表现者（完整数据集 - 20,265个样本）：**
1. **Qualifire** - 总体性能最佳，准确率94.57%，各项指标平衡优秀
2. **Vijil mBERT** - 强劲第二名，准确率93.38%，精确率高
3. **PreambleAI** - 高精确率（96.64%），总体性能良好

**不同数据集规模的一致性：**
- **Qualifire** 在所有数据集规模中都稳居第一
- **SavantAI** 和 **Palo Alto Networks** 在较小数据集上表现强劲
- **ProtectAI v1** 始终显示低召回率但高精确率
- **Deepset DeBERTa** 召回率极高但精确率较差（误报率高）

**权衡取舍：**
- **高精确率、低召回率**：PreambleAI、Llama Prompt Guard、ProtectAI v1
- **高召回率、低精确率**：Deepset DeBERTa
- **平衡性能**：Qualifire、Vijil mBERT、SavantAI

## 数据集管理

### 支持的输入格式
- **CSV**：逗号分隔值
- **TSV**：制表符分隔值
- **JSON**：JavaScript对象表示法
- **Parquet**：Apache Parquet格式

### 可用数据集

项目包含一个完整的合并数据集和三个预生成的平衡分布短数据集：

#### 数据集概览

| 数据集 | 目标样本数 | 实际样本数 | 安全(0) | 威胁(1) |
|--------|------------|------------|---------|---------|
| merged_datasets.csv | 完整 | 20,496 | 12,076 (58.9%) | 8,420 (41.1%) |
| merged_datasets_short_1000.csv | 1,036 | 1,000 | 515 (51.5%) | 485 (48.5%) |
| merged_datasets_short_2000.csv | 2,340 | 2,000 | 1,006 (50.3%) | 994 (49.7%) |
| merged_datasets_short_3000.csv | 3,840 | 3,000 | 1,569 (52.3%) | 1,431 (47.7%) |

#### 来源分布

| 来源数据集 | merged_datasets.csv | 1000.csv | 2000.csv | 3000.csv |
|------------|---------------------|----------|----------|----------|
| **总样本数** | **20,496** | **1,000** | **2,000** | **3,000** |
| qualifire_prompt-injections-benchmark_test | 5,000 (24.4%) | 116 (11.6%) | 260 (13.0%) | 427 (14.2%) |
| train-00000-of-00001 | 8,236 (40.2%) | 115 (11.5%) | 260 (13.0%) | 427 (14.2%) |
| test-00000-of-00001 | 2,060 (10.1%) | 115 (11.5%) | 260 (13.0%) | 426 (14.2%) |
| allenai_wildjailbreak-eval | 2,210 (10.8%) | 115 (11.5%) | 260 (13.0%) | 427 (14.2%) |
| jackhhao_jailbreak-classification-jailbreak_dataset_full | 1,998 (9.7%) | 115 (11.5%) | 260 (13.0%) | 427 (14.2%) |
| train-00000-of-00001-9564e8b05b4757ab | 546 (2.7%) | 115 (11.5%) | 260 (13.0%) | 426 (14.2%) |
| PurpleLlama_CybersecurityBenchmarks_prompt_injection | 251 (1.2%) | 115 (11.5%) | 245 (12.2%) | 245 (8.2%) |
| test-00000-of-00001-701d16158af87368 | 116 (0.6%) | 115 (11.5%) | 116 (5.8%) | 116 (3.9%) |
| rubend18_ChatGPT-Jailbreak-Prompts | 79 (0.4%) | 79 (7.9%) | 79 (4.0%) | 79 (2.6%) |

*merged_datasets.csv实际有20265个样本，231个样本是重复的，已被移除。*

#### 按来源的标签分布（merged_datasets.csv）

| 来源数据集 | 总计 | 安全(0) | 威胁(1) | 安全% | 威胁% |
|------------|------|---------|---------|-------|-------|
| qualifire_prompt-injections-benchmark_test | 5,000 | 3,001 | 1,999 | 60.0% | 40.0% |
| train-00000-of-00001 | 8,236 | 5,740 | 2,496 | 69.7% | 30.3% |
| test-00000-of-00001 | 2,060 | 1,410 | 650 | 68.4% | 31.6% |
| allenai_wildjailbreak-eval | 2,210 | 210 | 2,000 | 9.5% | 90.5% |
| jackhhao_jailbreak-classification-jailbreak_dataset_full | 1,998 | 1,332 | 666 | 66.7% | 33.3% |
| train-00000-of-00001-9564e8b05b4757ab | 546 | 343 | 203 | 62.8% | 37.2% |
| PurpleLlama_CybersecurityBenchmarks_prompt_injection | 251 | 0 | 251 | 0.0% | 100.0% |
| test-00000-of-00001-701d16158af87368 | 116 | 56 | 60 | 48.3% | 51.7% |
| rubend18_ChatGPT-Jailbreak-Prompts | 79 | 74 | 5 | 93.7% | 6.3% |

### 短数据集创建

`create_short_dataset.py`脚本允许您从完整合并数据集创建平衡的较小数据集，用于更快的测试和开发：

#### 功能特性
- **平衡采样**：保持每个来源数据集的比例代表性
- **标签平衡**：在可能的情况下保持每个来源内的标签分布
- **可配置大小**：指定目标样本数量
- **可重现**：使用随机种子确保一致结果
- **自动命名**：根据实际样本数量生成描述性文件名

#### 使用方法

```bash
cd datasets

# 基本用法 - 默认创建1000个样本
python create_short_dataset.py

# 自定义样本大小
python create_short_dataset.py --samples=1036
#python create_short_dataset.py --samples=2340
#python create_short_dataset.py --samples=3840

# 指定输入和输出文件
python create_short_dataset.py --input merged_datasets.csv --output my_dataset.csv --samples 500

# 使用自定义随机种子
python create_short_dataset.py --samples 1000 --seed 123
```

#### 命令行选项

| 选项 | 描述 | 默认值 |
|------|------|--------|
| `--input` | 输入CSV文件路径 | `merged_datasets.csv` |
| `--output` | 输出CSV文件路径（如未指定则自动生成） | 自动生成 |
| `--samples` | 目标提取样本数 | 1000 |
| `--seed` | 随机种子用于可重现性 | 42 |

#### 示例输出

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

### 数据集结构

合并后的所有数据集都遵循相同的结构：

| 列名 | 描述 | 类型 |
|------|------|------|
| text | 提示/文本内容 | 字符串 |
| label | 二元标签（0=安全，1=威胁） | 整数 |
| source | 来源数据集标识符 | 字符串 |
| original_label | 来源的原始标签 | 字符串 |

### 标签映射

合并过程将所有标签标准化为二元格式：
- **0（安全）**：正常、良性提示
- **1（威胁）**：提示注入尝试、越狱、恶意提示

## 模型配置

### 支持的模型
- **Llama Prompt Guard 2 (86M)**：`./models/Llama-Prompt-Guard-2-86M/`
- **ProtectAI DeBERTa v3**：`./models/protectaiv2/`
- **ProtectAI DeBERTa v1**：`./models/protectaiv1/`
- **PreambleAI**：`./models/preambleai/`
- **Qualifire**：`./models/qualifire/`
- **SavantAI**：`./models/testsavantai-prompt-injection-defender-large-v0/`
- **Deepset DeBERTa**：`./models/deepset-deberta/`
- **Vijil mBERT**：`./models/vijil-mbert-prompt-injection/`
- **您自己的模型...**

### 模型下载
```bash
# 使用huggingface-cli下载模型示例
huggingface-cli download PreambleAI/prompt-injection-defense --local-dir ./models/preambleai/
huggingface-cli download protectai/deberta-v3-base-prompt-injection-v2 --local-dir ./models/protectaiv2/
```

## 输出文件

结果保存在`results/`下的时间戳目录中：
```
results/evaluation_results_{model_name}_{timestamp}/
├── eval_results_{timestamp}.json      # 包含预测的详细结果
├── eval_summary_{timestamp}.csv       # 用于分析的扁平化结果
├── evaluation_report_{timestamp}.txt  # 人类可读报告
└── evaluation_metrics_{timestamp}.json # 结构化指标
```

## 评估指标

工具提供全面的评估指标：
- **准确率**：总体分类准确性
- **精确率**：威胁检测精确率
- **召回率**：威胁检测召回率（敏感性）
- **F1分数**：精确率和召回率的调和平均值
- **误报率（FPR）**：虚假威胁检测率
- **混淆矩阵**：详细的分类细分

## 高级用法

### 自定义模型集成
要添加您自己的模型，请在`models/`下创建目录，并确保它遵循HuggingFace transformers格式。

### 批量评估
```bash
# 使用同一数据集评估多个模型
for model in deepset-deberta testsavantai-prompt-injection-defender-large-v0 Llama-Prompt-Guard-2-86M protectaiv1 protectaiv2 preambleai qualifire; do
    python model-eval-v2.0.py --model ./models/$model/ --dataset datasets/merged_datasets_short_2000.csv
done
```

### 采样策略
```bash
# 从完整数据集随机采样
python model-eval-v2.0.py --model ./models/deepset-deberta/ --sample 500

# 使用预创建的平衡短数据集
python model-eval-v2.0.py --model ./models/deepset-deberta/ --dataset datasets/merged_datasets_short_1000.csv
```

## 版本历史

- **v2.0**：增强的评估管道，改进输出格式
- **v1.1**：添加采样支持和改进指标
- **v1.0**：具有基本模型评估的初始版本

## 模型来源

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
*未指定训练数据集*

## 数据集来源

### 4.1 Deepset
Qualifire用于训练的数据集：
https://huggingface.co/datasets/deepset/prompt-injections

testsavantai/prompt-injection-defender-large-v0使用的此数据集克隆：
https://huggingface.co/datasets/JasperLS/prompt-injections

### 4.2 allenai/wildjailbreak
Qualifire用于训练的数据集：
https://huggingface.co/datasets/allenai/wildjailbreak

### 4.3 jackhhao/jailbreak-classification
ProtectAI和Qualifire用于训练的数据集：
https://huggingface.co/datasets/jackhhao/jailbreak-classification

### 4.4 qualifire/prompt-injections-benchmark
https://huggingface.co/datasets/qualifire/prompt-injections-benchmark

### 4.5 xTRam1/safe-guard-prompt-injection
https://huggingface.co/datasets/xTRam1/safe-guard-prompt-injection

### 4.6 rubend18/ChatGPT-Jailbreak-Prompts
ProtectAI和testsavantai/prompt-injection-defender-large-v0用于训练的数据集：
https://huggingface.co/datasets/rubend18/ChatGPT-Jailbreak-Prompts

### 4.7 NVIDIA nvidia/Aegis-AI-Content-Safety-Dataset-2.0
Prisma AIRS用于训练的数据集：
https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0

### 4.8 Purple Llama CyberSecEval
Llama Prompt Guard用于训练的数据集：
https://github.com/meta-llama/PurpleLlama/tree/23156b70efb596831c02c6461fc42da1f75988ec/CybersecurityBenchmarks

## 许可证

本项目是开源的。请查看各个模型许可证了解其各自的使用条款。