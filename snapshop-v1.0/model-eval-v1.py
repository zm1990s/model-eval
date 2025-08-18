# Don't delete the following comment, it is used by the system to identify the file.
# Supported models:

# meta-llama/Prompt-Guard-86M
# https://huggingface.co/meta-llama/Prompt-Guard-86M/tree/main

# protectai/deberta-v3-base-prompt-injection-v2
# https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2/tree/main

# protectai/deberta-v3-base-prompt-injection
# https://huggingface.co/protectai/deberta-v3-base-prompt-injection/tree/main

# PreambleAI/prompt-injection-defense
# https://huggingface.co/PreambleAI/prompt-injection-defense/tree/main

# qualifire/prompt-injection-sentinel
# https://huggingface.co/qualifire/prompt-injection-sentinel/tree/main

# testsavantai/prompt-injection-defender-large-v0
# https://huggingface.co/testsavantai/prompt-injection-defender-large-v0/tree/main

# Not supported: 
# vijil/mbert-prompt-injection
# https://huggingface.co/vijil/mbert-prompt-injection/tree/main

# conda create -n model-eval python=3.12
# conda activate model-eval
# python -m pip install torch pandas transformers tqdm pyarrow -i https://pypi.tuna.tsinghua.edu.cn/simple 
# huggingface-cli download PreambleAI/prompt-injection-defense --local-dir ./models/preambleai/

import pandas as pd
import json
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from tqdm import tqdm
import os
import torch
import numpy as np

# 模型配置
#model_path = "./models/Llama-Prompt-Guard-2-86M/"
#model_path = "./models/preambleai/"
#model_path = "./models/qualifire/" 
model_path = "./models/protectaiv2/"
#model_path = "./models/protectaiv1/"
#model_path = "./models/testsavantai-prompt-injection-defender-large-v0/"
datasets_path = "./datasets/"

print(f"使用本地模型: {model_path}")


tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path, use_safetensors=True)

# 检查模型类型
model_type = model.config.model_type if hasattr(model.config, 'model_type') else 'unknown'
print(f"检测到模型类型: {model_type}")

# 提取模型名称
def get_model_name(model_path):
    """从模型路径中提取模型名称"""
    # 移除末尾的斜杠
    model_path = model_path.rstrip('/')
    # 获取最后一个目录名
    model_name = os.path.basename(model_path)
    # 如果是相对路径，可能会有 . 开头，需要进一步处理
    if model_name.startswith('.'):
        # 处理 ./models/xxx 的情况
        parts = model_path.split('/')
        model_name = parts[-1] if len(parts) > 1 else model_name
    
    # 清理模型名称，移除特殊字符，替换为下划线
    import re
    model_name = re.sub(r'[^\w\-.]', '_', model_name)
    return model_name

# 自定义预测函数来处理token_type_ids问题
def safe_predict(text):
    """安全的预测函数，处理token_type_ids问题"""
    try:
        # 手动tokenize并移除token_type_ids
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512,
            padding=True,
            return_token_type_ids=False  # 明确不返回token_type_ids
        )
        
        # 直接使用模型进行预测
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # 格式化输出以匹配pipeline格式
        labels = model.config.id2label if hasattr(model.config, 'id2label') else {0: "BENIGN", 1: "INJECTION"}
        results = []
        for i, score in enumerate(predictions[0]):
            results.append({
                "label": labels.get(i, f"LABEL_{i}"),
                "score": float(score)
            })
        
        return results
        
    except Exception as e:
        print(f"预测时出错: {e}")
        return None

def get_predicted_label(prediction_result):
    """从预测结果中获取预测标签（0或1）"""
    if not prediction_result:
        return None
    
    # 获取最高置信度的预测
    top_pred = max(prediction_result, key=lambda x: x['score'])
    label_str = top_pred['label']
    
    # 根据标签字符串映射到数值
    if label_str in ['SAFE', 'BENIGN', 'LABEL_0', 'trusted','benign']:
        return 0
    elif label_str in ['INJECTION', 'UNSAFE', 'LABEL_1','untrusted','jailbreak']:
        return 1
    else:
        # 如果标签包含数字，直接提取数字
        import re
        number_match = re.search(r'LABEL_(\d+)', label_str)
        if number_match:
            return int(number_match.group(1))
        
        # 如果标签不在预期范围内，尝试从预测结果中找到对应的标签
        for pred in prediction_result:
            if 'INJECTION' in pred['label'] or 'UNSAFE' in pred['label'] or 'LABEL_1' in pred['label']:
                if pred == top_pred:
                    return 1
        return 0

def calculate_metrics(true_labels, predicted_labels):
    """计算评估指标"""
    # 确保输入是numpy数组
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    
    # 计算混淆矩阵的各个值
    tp = np.sum((true_labels == 1) & (predicted_labels == 1))  # True Positive
    tn = np.sum((true_labels == 0) & (predicted_labels == 0))  # True Negative
    fp = np.sum((true_labels == 0) & (predicted_labels == 1))  # False Positive
    fn = np.sum((true_labels == 1) & (predicted_labels == 0))  # False Negative
    
    # 计算各种指标
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision,
        'fpr': fpr,
        'f1': f1
    }

def create_output_directory():
    """创建输出文件夹，包含模型名称"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = get_model_name(model_path)
    output_dir = f"evaluation_results_{model_name}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def calculate_metrics_from_csv(csv_file_path):
    """从CSV文件中读取数据并计算评估指标"""
    try:
        # 读取CSV文件
        df = pd.read_csv(csv_file_path)
        print(f"\n=== 从CSV文件计算评估指标 ===")
        print(f"读取CSV文件: {csv_file_path}")
        print(f"CSV文件包含 {len(df)} 行数据")
        
        # 过滤有效数据（有原始标签和预测标签的）
        valid_data = df.dropna(subset=['original_label', 'predicted_label'])
        print(f"有效数据行数: {len(valid_data)}")
        
        if len(valid_data) == 0:
            print("警告: 没有找到有效的标签数据")
            return None
        
        # 获取真实标签和预测标签
        true_labels = valid_data['original_label'].values
        predicted_labels = valid_data['predicted_label'].values
        
        # 计算指标
        metrics = calculate_metrics(true_labels, predicted_labels)
        
        # 统计prediction_result列的分布
        if 'prediction_result' in df.columns:
            result_counts = df['prediction_result'].value_counts()
            print(f"\n预测结果分布:")
            for result_type, count in result_counts.items():
                print(f"  {result_type}: {count}")
        
        return metrics, valid_data
        
    except Exception as e:
        print(f"从CSV计算指标时出错: {e}")
        return None

def save_evaluation_results(metrics, csv_file_path, valid_data_count, output_dir):
    """保存评估结果到文件"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 保存详细的评估报告
    report_file = os.path.join(output_dir, f"evaluation_report_{timestamp}.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("模型评估报告\n")
        f.write("=" * 50 + "\n")
        f.write(f"评估时间: {datetime.now().isoformat()}\n")
        f.write(f"数据来源: {csv_file_path}\n")
        f.write(f"有效样本数: {valid_data_count}\n\n")
        
        f.write("混淆矩阵:\n")
        f.write("-" * 20 + "\n")
        f.write(f"True Positive (TP):  {metrics['tp']}\n")
        f.write(f"True Negative (TN):  {metrics['tn']}\n")
        f.write(f"False Positive (FP): {metrics['fp']}\n")
        f.write(f"False Negative (FN): {metrics['fn']}\n\n")
        
        f.write("评估指标:\n")
        f.write("-" * 20 + "\n")
        f.write(f"准确度 (Accuracy):    {metrics['accuracy']:.4f}\n")
        f.write(f"召回率 (Recall):      {metrics['recall']:.4f}\n")
        f.write(f"精确度 (Precision):   {metrics['precision']:.4f}\n")
        f.write(f"误报率 (FPR):         {metrics['fpr']:.4f}\n")
        f.write(f"F1分数 (F1):          {metrics['f1']:.4f}\n\n")
        
        f.write("指标说明:\n")
        f.write("-" * 20 + "\n")
        f.write("准确度 (Accuracy) = (TP + TN) / (TP + TN + FP + FN)\n")
        f.write("召回率 (Recall) = TP / (TP + FN)\n")
        f.write("精确度 (Precision) = TP / (TP + FP)\n")
        f.write("误报率 (FPR) = FP / (FP + TN)\n")
        f.write("F1分数 (F1) = 2 × (Precision × Recall) / (Precision + Recall)\n")
    
    print(f"评估报告已保存到: {report_file}")
    
    # 保存JSON格式的指标
    metrics_file = os.path.join(output_dir, f"evaluation_metrics_{timestamp}.json")
    metrics_data = {
        'evaluation_time': datetime.now().isoformat(),
        'data_source': csv_file_path,
        'total_valid_samples': valid_data_count,
        'confusion_matrix': {
            'true_positive': metrics['tp'],
            'true_negative': metrics['tn'],
            'false_positive': metrics['fp'],
            'false_negative': metrics['fn']
        },
        'metrics': {
            'accuracy': metrics['accuracy'],
            'recall': metrics['recall'],
            'precision': metrics['precision'],
            'false_positive_rate': metrics['fpr'],
            'f1_score': metrics['f1']
        },
        'formulas': {
            'accuracy': '(TP + TN) / (TP + TN + FP + FN)',
            'recall': 'TP / (TP + FN)',
            'precision': 'TP / (TP + FP)',
            'false_positive_rate': 'FP / (FP + TN)',
            'f1_score': '2 × (Precision × Recall) / (Precision + Recall)'
        }
    }
    
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics_data, f, ensure_ascii=False, indent=2)
    
    print(f"评估指标JSON已保存到: {metrics_file}")

def process_dataset(file_path, dataset_name):
    """处理单个数据集文件"""
    print(f"\n处理数据集: {dataset_name}")
    print(f"文件路径: {file_path}")
    
    # 读取parquet文件
    try:
        df = pd.read_parquet(file_path)
        print(f"数据集大小: {len(df)} 行")
        print(f"列名: {list(df.columns)}")
        
        # 显示前几行数据结构
        print(f"\n前3行数据:")
        print(df.head(3))
        
    except Exception as e:
        print(f"读取文件错误: {e}")
        return None
    
    results = []
    
    # 假设文本列可能叫 'text', 'prompt', 'input' 等
    text_columns = [col for col in df.columns if any(keyword in col.lower() 
                   for keyword in ['text', 'prompt', 'input', 'content', 'message'])]
    
    if not text_columns:
        # 如果没找到明显的文本列，使用第一列
        text_column = df.columns[0]
        print(f"警告: 未找到明显的文本列，使用第一列: {text_column}")
    else:
        text_column = text_columns[0]
        print(f"使用文本列: {text_column}")
    
    # 查找标签列
    label_columns = [col for col in df.columns if any(keyword in col.lower() 
                    for keyword in ['label', 'target', 'ground_truth', 'gt'])]
    
    if label_columns:
        label_column = label_columns[0]
        print(f"使用标签列: {label_column}")
    else:
        label_column = None
        print("警告: 未找到标签列")
    
    # 处理每一行
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"处理{dataset_name}"):
        try:
            text = str(row[text_column])
            
            # 跳过空文本
            if not text or text.strip() == '':
                print(f"跳过空文本，行索引: {idx}")
                continue
            
            # 使用安全的预测函数
            prediction = safe_predict(text)
            
            # 获取原始标签
            original_label = row[label_column] if label_column else None
            
            # 获取预测标签
            predicted_label = get_predicted_label(prediction)
            
            # 记录结果
            result = {
                'dataset': dataset_name,
                'index': idx,
                'text': text,
                'prediction': prediction,
                'original_label': original_label,
                'predicted_label': predicted_label,
                'timestamp': datetime.now().isoformat()
            }
            
            # 添加原始数据的其他列
            for col in df.columns:
                if col != text_column and col != label_column:
                    result[f'original_{col}'] = row[col]
            
            results.append(result)
            
        except Exception as e:
            print(f"处理第{idx}行时出错: {e}")
            results.append({
                'dataset': dataset_name,
                'index': idx,
                'text': str(row[text_column]) if text_column in row else 'N/A',
                'prediction': None,
                'original_label': row[label_column] if label_column else None,
                'predicted_label': None,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
    
    return results

def main():
    # 创建输出文件夹
    output_dir = create_output_directory()
    print(f"\n输出文件夹: {output_dir}")
    
    # 测试模型是否正常工作
    print("\n=== 测试模型 ===")
    test_text = "Your prompt injection is here"
    test_result = safe_predict(test_text)
    print(f"测试文本: {test_text}")
    print(f"测试结果: {test_result}")
    
    if test_result is None:
        print("模型测试失败，请检查模型配置")
        return
    
    print("\n=== 模型信息 ===")
    print(f"模型类型: {model_type}")
    print(f"模型配置: {model.config}")
    if hasattr(model.config, 'id2label'):
        print(f"标签映射: {model.config.id2label}")
    
    # 数据集文件路径
    train_file = datasets_path + "train-00000-of-00001-9564e8b05b4757ab.parquet"
    test_file = datasets_path + "test-00000-of-00001-701d16158af87368.parquet"
    
    all_results = []
    
    # 处理训练集
    if os.path.exists(train_file):
        train_results = process_dataset(train_file, "train")
        if train_results:
            all_results.extend(train_results)
    else:
        print(f"训练文件不存在: {train_file}")
    
    # 处理测试集
    if os.path.exists(test_file):
        test_results = process_dataset(test_file, "test")
        if test_results:
            all_results.extend(test_results)
    else:
        print(f"测试文件不存在: {test_file}")
    
    # 保存结果
    if all_results:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存为JSON文件
        output_file = os.path.join(output_dir, f"eval_results_{timestamp}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {output_file}")
        
        # 创建结果摘要
        df_results = pd.DataFrame(all_results)
        
        # 统计分析
        print(f"\n=== 结果摘要 ===")
        print(f"总处理条目: {len(all_results)}")
        
        # 按数据集统计
        dataset_counts = df_results['dataset'].value_counts()
        print(f"各数据集处理条目数:")
        for dataset, count in dataset_counts.items():
            print(f"  {dataset}: {count}")
        
        # 分类结果统计（如果有成功的预测）
        successful_predictions = [r for r in all_results if r.get('prediction') is not None and r.get('predicted_label') is not None]
        if successful_predictions:
            print(f"\n成功预测条目: {len(successful_predictions)}")
            
            # 统计各类别预测结果
            label_counts = {}
            for result in successful_predictions:
                if result['prediction']:
                    top_prediction = max(result['prediction'], key=lambda x: x['score'])
                    label = top_prediction['label']
                    label_counts[label] = label_counts.get(label, 0) + 1
            
            print("预测标签分布:")
            for label, count in sorted(label_counts.items()):
                print(f"  {label}: {count}")
                
            # 计算平均置信度
            total_confidence = sum(max(r['prediction'], key=lambda x: x['score'])['score'] 
                                 for r in successful_predictions if r['prediction'])
            avg_confidence = total_confidence / len(successful_predictions)
            print(f"\n平均置信度: {avg_confidence:.4f}")
            
            # 计算评估指标
            valid_results = [r for r in successful_predictions if r.get('original_label') is not None]
            if valid_results:
                true_labels = [r['original_label'] for r in valid_results]
                predicted_labels = [r['predicted_label'] for r in valid_results]
                
                metrics = calculate_metrics(true_labels, predicted_labels)
                
                print(f"\n=== 评估指标 ===")
                print(f"True Positive (TP): {metrics['tp']}")
                print(f"True Negative (TN): {metrics['tn']}")
                print(f"False Positive (FP): {metrics['fp']}")
                print(f"False Negative (FN): {metrics['fn']}")
                print(f"准确度 (Accuracy): {metrics['accuracy']:.4f}")
                print(f"召回率 (Recall): {metrics['recall']:.4f}")
                print(f"精确度 (Precision): {metrics['precision']:.4f}")
                print(f"误报率 (FPR): {metrics['fpr']:.4f}")
                print(f"F1分数 (F1): {metrics['f1']:.4f}")
        
        # 保存增强的摘要CSV
        summary_file = os.path.join(output_dir, f"eval_summary_{timestamp}.csv")
        
        # 创建用于CSV的扁平化数据
        flattened_results = []
        for result in all_results:
            flat_result = {
                'dataset': result['dataset'],
                'index': result['index'],
                'text': result['text'][:200] + '...' if len(result['text']) > 200 else result['text'],  # 截断长文本
                'original_label': result.get('original_label'),
                'predicted_label': result.get('predicted_label'),
                'timestamp': result['timestamp']
            }
            
            # 添加预测vs真实对比
            if result.get('original_label') is not None and result.get('predicted_label') is not None:
                true_label = result['original_label']
                pred_label = result['predicted_label']
                
                if true_label == 1 and pred_label == 1:
                    flat_result['prediction_result'] = 'TP'  # True Positive
                elif true_label == 0 and pred_label == 0:
                    flat_result['prediction_result'] = 'TN'  # True Negative
                elif true_label == 0 and pred_label == 1:
                    flat_result['prediction_result'] = 'FP'  # False Positive
                elif true_label == 1 and pred_label == 0:
                    flat_result['prediction_result'] = 'FN'  # False Negative
                else:
                    flat_result['prediction_result'] = 'Unknown'
            else:
                flat_result['prediction_result'] = 'N/A'
            
            if result.get('prediction'):
                # 获取最高置信度的预测
                top_pred = max(result['prediction'], key=lambda x: x['score'])
                flat_result['predicted_label_text'] = top_pred['label']
                flat_result['confidence_score'] = top_pred['score']
                
                # 添加所有预测分数
                for pred in result['prediction']:
                    flat_result[f"score_{pred['label']}"] = pred['score']
            else:
                flat_result['predicted_label_text'] = 'ERROR'
                flat_result['confidence_score'] = 0.0
                flat_result['error'] = result.get('error', 'Unknown error')
            
            flattened_results.append(flat_result)
        
        df_summary = pd.DataFrame(flattened_results)
        df_summary.to_csv(summary_file, index=False, encoding='utf-8')
        print(f"结果摘要已保存到: {summary_file}")
        
        # 从CSV文件计算并保存评估指标
        csv_metrics_result = calculate_metrics_from_csv(summary_file)
        if csv_metrics_result:
            metrics, valid_data = csv_metrics_result
            
            print(f"\n=== 基于CSV的评估指标 ===")
            print(f"True Positive (TP): {metrics['tp']}")
            print(f"True Negative (TN): {metrics['tn']}")
            print(f"False Positive (FP): {metrics['fp']}")
            print(f"False Negative (FN): {metrics['fn']}")
            print(f"准确度 (Accuracy): {metrics['accuracy']:.4f}")
            print(f"召回率 (Recall): {metrics['recall']:.4f}")
            print(f"精确度 (Precision): {metrics['precision']:.4f}")
            print(f"误报率 (FPR): {metrics['fpr']:.4f}")
            print(f"F1分数 (F1): {metrics['f1']:.4f}")
            
            # 保存评估结果到文件
            save_evaluation_results(metrics, summary_file, len(valid_data), output_dir)
        
        # 显示一些样例结果
        print(f"\n=== 样例结果 ===")
        for i, result in enumerate(all_results[:3]):
            print(f"\n样例 {i+1}:")
            print(f"文本: {result['text'][:100]}...")
            print(f"原始标签: {result.get('original_label')}")
            if result['prediction']:
                top_pred = max(result['prediction'], key=lambda x: x['score'])
                print(f"预测: {top_pred['label']} (置信度: {top_pred['score']:.4f})")
                print(f"预测标签: {result.get('predicted_label')}")
            else:
                print(f"预测失败: {result.get('error', 'Unknown error')}")
    
    else:
        print("没有处理任何数据")

if __name__ == "__main__":
    main()