import pandas as pd
import json
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from tqdm import tqdm
import os
import torch
import numpy as np

# 模型配置
model_path = "./models/vijil-mbert-prompt-injection/"
datasets_path = "./datasets/"

print(f"使用本地模型: {model_path}")

# 加载模型
tokenizer = AutoTokenizer.from_pretrained(model_path) 
model = AutoModelForSequenceClassification.from_pretrained(model_path)

classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    truncation=True,
    max_length=512,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)

# 创建输出文件夹
def create_output_directory():
    """创建以模型名称和时间戳命名的输出文件夹"""
    model_name = "vijil"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"evaluation_results_{model_name}_{timestamp}"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    return output_dir

def safe_predict(text):
    """安全预测函数，处理异常情况"""
    try:
        if not text or pd.isna(text):
            return None
        
        text = str(text).strip()
        if not text:
            return None
            
        result = classifier(text)
        return result[0] if result else None
    except Exception as e:
        print(f"预测错误: {e}")
        return None

def get_predicted_label(prediction_result):
    """从预测结果中获取预测标签（0或1）"""
    if not prediction_result:
        return None
    
    label = prediction_result['label']
    
    # 如果已经是数字，直接返回
    if isinstance(label, (int, float)):
        return int(label)
    
    # 转换为字符串处理
    label_str = str(label).lower()
    
    # 根据标签字符串映射到数值
    if label_str in ['safe', 'benign', 'label_0', 'trusted', 'benign', 'legitimate', '0']:
        return 0
    elif label_str in ['injection', 'unsafe', 'label_1', 'untrusted', 'jailbreak', 'injection', '1']:
        return 1
    else:
        # 如果标签包含数字，直接提取数字
        import re
        number_match = re.search(r'label_(\d+)', label_str)
        if number_match:
            return int(number_match.group(1))
        
        # 对于vijil模型，通常标签为 'injection' 或 'legitimate'
        if 'injection' in label_str:
            return 1
        elif 'legitimate' in label_str:
            return 0
        
        return 0  # 默认为benign

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
    
    # 查找文本列和标签列
    text_columns = [col for col in df.columns if any(keyword in col.lower() 
                   for keyword in ['text', 'prompt', 'input', 'content', 'message'])]
    
    label_columns = [col for col in df.columns if any(keyword in col.lower() 
                    for keyword in ['label', 'target', 'class', 'category'])]
    
    if not text_columns:
        print("未找到文本列，尝试使用第一列作为文本列")
        text_column = df.columns[0]
    else:
        text_column = text_columns[0]
        print(f"使用文本列: {text_column}")
    
    if not label_columns:
        print("未找到标签列，尝试使用第二列作为标签列")
        if len(df.columns) > 1:
            label_column = df.columns[1]
        else:
            label_column = None
            print("警告：没有找到标签列")
    else:
        label_column = label_columns[0]
        print(f"使用标签列: {label_column}")
    
    # 开始预测
    print(f"\n开始预测...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"处理 {dataset_name}"):
        text = row[text_column]
        original_label = row[label_column] if label_column else None
        
        # 进行预测
        prediction_result = safe_predict(text)
        
        # 记录结果
        result_entry = {
            'dataset': dataset_name,
            'index': idx,
            'text': text,
            'original_label': original_label,
            'prediction': prediction_result,
            'predicted_label': prediction_result['label'] if prediction_result else None,
            'predicted_score': prediction_result['score'] if prediction_result else None,
        }
        
        results.append(result_entry)
    
    return results

def calculate_metrics_from_csv(csv_file_path):
    """从CSV文件中读取数据并计算评估指标"""
    try:
        # 读取CSV文件
        df = pd.read_csv(csv_file_path)
        print(f"\n=== 从CSV文件计算评估指标 ===")
        print(f"读取CSV文件: {csv_file_path}")
        print(f"CSV文件包含 {len(df)} 行数据")
        
        # 过滤有效数据（有原始标签和预测标签的）
        valid_data = df.dropna(subset=['original_label', 'predicted_label_binary'])
        print(f"有效数据行数: {len(valid_data)}")
        
        if len(valid_data) == 0:
            print("警告: 没有找到有效的标签数据")
            return None
        
        # 获取真实标签和预测标签
        true_labels = valid_data['original_label'].values
        predicted_labels = valid_data['predicted_label_binary'].values
        
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

def save_results_to_csv(results, output_dir):
    """将结果保存为CSV文件"""
    if not results:
        print("没有结果可保存")
        return
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 创建用于CSV的扁平化数据
    flattened_results = []
    for result in results:
        flat_result = {
            'dataset': result['dataset'],
            'index': result['index'],
            'text': result['text'][:200] + '...' if len(result['text']) > 200 else result['text'],  # 截断长文本
            'original_label': result.get('original_label'),
            'predicted_label': result.get('predicted_label'),
            'predicted_score': result.get('predicted_score')
        }
        
        # 获取二进制预测标签
        prediction_result = result.get('prediction')
        predicted_label_binary = get_predicted_label(prediction_result)
        flat_result['predicted_label_binary'] = predicted_label_binary
        
        # 添加预测vs真实对比
        if result.get('original_label') is not None and predicted_label_binary is not None:
            true_label = result['original_label']
            pred_label = predicted_label_binary
            
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
        
        flattened_results.append(flat_result)
    
    df_summary = pd.DataFrame(flattened_results)
    csv_file = os.path.join(output_dir, f"eval_summary_{timestamp}.csv")
    df_summary.to_csv(csv_file, index=False, encoding='utf-8')
    print(f"结果摘要已保存到: {csv_file}")
    
    return csv_file

def generate_summary_report(results, output_dir):
    """生成汇总报告"""
    if not results:
        return
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 创建DataFrame用于分析
    df_results = pd.DataFrame(results)
    
    # 生成汇总统计
    summary = {
        'total_predictions': len(results),
        'successful_predictions': len([r for r in results if r['prediction'] is not None]),
        'failed_predictions': len([r for r in results if r['prediction'] is None]),
        'datasets_processed': df_results['dataset'].nunique(),
        'prediction_distribution': {},
        'dataset_distribution': df_results['dataset'].value_counts().to_dict()
    }
    
    # 统计预测标签分布
    successful_predictions = df_results[df_results['predicted_label'].notna()]
    if not successful_predictions.empty:
        summary['prediction_distribution'] = successful_predictions['predicted_label'].value_counts().to_dict()
    
    # 计算评估指标（如果有标签数据）
    valid_results = [r for r in results if r.get('original_label') is not None and r.get('prediction') is not None]
    if valid_results:
        true_labels = [r['original_label'] for r in valid_results]
        predicted_labels = [get_predicted_label(r['prediction']) for r in valid_results]
        
        # 过滤掉预测失败的数据
        valid_pairs = [(t, p) for t, p in zip(true_labels, predicted_labels) if p is not None]
        if valid_pairs:
            true_labels_clean = [pair[0] for pair in valid_pairs]
            predicted_labels_clean = [pair[1] for pair in valid_pairs]
            
            metrics = calculate_metrics(true_labels_clean, predicted_labels_clean)
            summary['evaluation_metrics'] = metrics
    
    # 保存汇总报告为JSON
    summary_file = os.path.join(output_dir, f"summary_report_{timestamp}.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"汇总报告已保存: {summary_file}")
    
    # 打印汇总信息
    print("\n=== 预测结果汇总 ===")
    print(f"总预测条目: {summary['total_predictions']}")
    print(f"成功预测: {summary['successful_predictions']}")
    print(f"失败预测: {summary['failed_predictions']}")
    print(f"处理的数据集数量: {summary['datasets_processed']}")
    
    if summary['prediction_distribution']:
        print("\n预测标签分布:")
        for label, count in summary['prediction_distribution'].items():
            print(f"  {label}: {count}")
    
    print("\n各数据集处理条目数:")
    for dataset, count in summary['dataset_distribution'].items():
        print(f"  {dataset}: {count}")
    
    # 如果有评估指标，打印出来
    if 'evaluation_metrics' in summary:
        metrics = summary['evaluation_metrics']
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
    print(f"模型路径: {model_path}")
    print(f"模型配置: {model.config}")
    if hasattr(model.config, 'id2label'):
        print(f"标签映射: {model.config.id2label}")
    
    # 数据集文件路径
    train_file = datasets_path + "train-00000-of-00001-9564e8b05b4757ab.parquet"
    test_file = datasets_path + "test-00000-of-00001-701d16158af87368.parquet"
    
    all_results = []
    
    # 处理训练集
    if os.path.exists(train_file):
        print(f"\n处理训练集: {train_file}")
        train_results = process_dataset(train_file, "train")
        if train_results:
            all_results.extend(train_results)
    else:
        print(f"训练文件不存在: {train_file}")
    
    # 处理测试集
    if os.path.exists(test_file):
        print(f"\n处理测试集: {test_file}")
        test_results = process_dataset(test_file, "test")
        if test_results:
            all_results.extend(test_results)
    else:
        print(f"测试文件不存在: {test_file}")
    
    # 保存结果
    if all_results:
        # 保存增强的摘要CSV
        csv_file = save_results_to_csv(all_results, output_dir)
        
        # 保存为JSON文件（详细结果）
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        json_file = os.path.join(output_dir, f"eval_results_{timestamp}.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"详细结果已保存到JSON文件: {json_file}")
        
        # 从CSV文件计算并保存评估指标
        csv_metrics_result = calculate_metrics_from_csv(csv_file)
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
            save_evaluation_results(metrics, csv_file, len(valid_data), output_dir)
        
        # 生成汇总报告
        generate_summary_report(all_results, output_dir)
        
        print(f"\n✅ 所有结果已保存到文件夹: {output_dir}")
    else:
        print("❌ 没有处理任何数据")

if __name__ == "__main__":
    main()