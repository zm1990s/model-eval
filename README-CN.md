# 提示注入检测模型评估工具

一个用于评估提示注入检测模型性能的综合工具。该脚本可以在提示注入数据集上评估模型性能，并生成详细的分析报告。

## 功能特性

- 支持多种模型类型（Llama Prompt Guard、ProtectAI DeBERTa 等）
- 自动处理 Parquet 格式数据集
- 生成全面的评估指标
- 创建包含模型名称和时间戳的有序输出目录
- 自动处理各种文本列格式
- 提供详细的混淆矩阵分析
- 导出多种格式的结果（JSON、CSV、TXT）

## 环境要求

### 环境设置
```bash
conda create -n model-eval python=3.12
conda activate model-eval
python -m pip install torch pandas transformers tqdm pyarrow
```

### 依赖包
- torch
- pandas
- transformers
- tqdm
- pyarrow
- numpy

## 项目结构

```
model-eval/
├── model-eval-v1.py          # 主评估脚本
├── models/                   # 模型目录
│   ├── Llama-Prompt-Guard-2-86M/
│   └── protectai/
├── datasets/                 # 数据集目录
│   ├── test-00000-of-00001-701d16158af87368.parquet
│   └── train-00000-of-00001-9564e8b05b4757ab.parquet
└── evaluation_results_*/     # 生成的输出目录
```

## 配置

### 模型配置
在 `model-eval-v1.py` 中编辑模型路径：

```python
# Llama Prompt Guard 模型
model_path = "./models/Llama-Prompt-Guard-2-86M/"

# ProtectAI DeBERTa 模型
model_path = "./models/protectai/"
```

### 数据集配置
```python
datasets_path = "./datasets/"
```

## 使用方法

### 基本用法
```bash
python model-eval-v1.py
```

### 支持的模型
- **Llama Prompt Guard 2 (86M)**: Meta 的提示注入检测模型
- **ProtectAI DeBERTa v3**: ProtectAI 基于 DeBERTa 的提示注入检测器

## 数据集格式

脚本自动检测和处理：
- **文本列**: `text`、`prompt`、`input`、`content`、`message`
- **标签列**: `label`、`target`、`ground_truth`、`gt`
- **格式**: 标准结构的 Parquet 文件

## 输出文件

脚本创建包含模型名称的时间戳目录：`evaluation_results_{模型名称}_{时间戳}/`

### 生成的文件

1. **`eval_results_{时间戳}.json`**
   - JSON 格式的完整评估结果
   - 包含所有预测、原始标签和元数据

2. **`eval_summary_{时间戳}.csv`**
   - 适合电子表格分析的扁平化结果
   - 包含预测比较（TP/TN/FP/FN）
   - 每个预测的置信度分数

3. **`evaluation_report_{时间戳}.txt`**
   - 人类可读的评估报告
   - 详细的混淆矩阵
   - 带解释的所有评估指标

4. **`evaluation_metrics_{时间戳}.json`**
   - JSON 格式的结构化评估指标
   - 包含公式和元数据

### 评估指标

- **准确度 (Accuracy)**: 整体正确率
- **召回率 (Recall/Sensitivity)**: 真正例率
- **精确度 (Precision)**: 正预测值
- **误报率 (FPR)**: 假警报率
- **F1 分数**: 精确度和召回率的调和平均数

### 混淆矩阵元素
- **TP (真正例)**: 正确识别的注入
- **TN (真负例)**: 正确识别的良性提示
- **FP (假正例)**: 被分类为注入的良性提示
- **FN (假负例)**: 遗漏的注入尝试

## 输出示例结构

```
evaluation_results_Llama-Prompt-Guard-2-86M_20250804_143022/
├── eval_results_20250804_143022.json
├── eval_summary_20250804_143022.csv
├── evaluation_report_20250804_143022.txt
└── evaluation_metrics_20250804_143022.json
```

## 故障排除

### 常见问题

1. **模型加载错误**
   - 确保模型文件已正确下载
   - 检查模型路径配置
   - 验证 safetensors 格式兼容性

2. **数据集格式问题**
   - 验证 Parquet 文件完整性
   - 检查列命名约定
   - 确保正确的文本编码

3. **内存问题**
   - 考虑批量处理数据集
   - 监控 GPU/CPU 内存使用
   - 如需要可减少 max_length 参数

## 模型来源

- [Llama Prompt Guard 86M](https://huggingface.co/meta-llama/Prompt-Guard-86M)
- [ProtectAI DeBERTa v3](https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2)

## 许可证

此评估工具用于研究和评估目的。请检查各个模型的许可证以了解使用限制。