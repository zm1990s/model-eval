"""
Model Evaluation Script for Vijil Model
Based on model-eval-v2.0.py methodology with vijil-specific adaptations.
"""

import pandas as pd
import json
import os
import glob
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from tqdm import tqdm
import torch
import numpy as np
import re
import argparse

class VijilModelEvaluator:
    def __init__(self, model_path=None, merged_dataset_path=None):
        """
        Initialize the Vijil Model Evaluator
        
        Args:
            model_path (str): Path to the vijil model
            merged_dataset_path (str): Path to merged dataset CSV file
        """
        # Default model path for vijil
        self.model_path = model_path or "./models/vijil-mbert-prompt-injection/"
        self.merged_dataset_path = merged_dataset_path
        
        print(f"Using local model: {self.model_path}")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        
        # Create classifier pipeline
        self.classifier = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            truncation=True,
            max_length=512,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        
        # Find dataset if not specified
        if not self.merged_dataset_path:
            self.merged_dataset_path = self.find_latest_merged_dataset()
        
        print(f"Using dataset: {self.merged_dataset_path}")
    
    def find_latest_merged_dataset(self):
        """Find the latest merged dataset file"""
        dataset_pattern = "./datasets/merged_datasets*.csv"
        csv_files = glob.glob(dataset_pattern)
        if csv_files:
            # Sort by modification time and return the latest
            latest_file = max(csv_files, key=os.path.getmtime)
            return latest_file
        else:
            return "./datasets/merged_datasets.csv"
    
    def get_model_name(self):
        """Extract model name from model path"""
        return "vijil"

    # Keep original predict functions unchanged as requested
    def safe_predict(self, text):
        """安全预测函数，处理异常情况"""
        try:
            if not text or pd.isna(text):
                return None
            
            text = str(text).strip()
            if not text:
                return None
                
            result = self.classifier(text)
            return result[0] if result else None
        except Exception as e:
            print(f"预测错误: {e}")
            return None

    def get_predicted_label(self, prediction_result):
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

    def calculate_metrics(self, true_labels, predicted_labels):
        """
        Calculate evaluation metrics
        
        Args:
            true_labels (array): Ground truth labels
            predicted_labels (array): Predicted labels
            
        Returns:
            dict: Dictionary containing various metrics
        """
        true_labels = np.array(true_labels)
        predicted_labels = np.array(predicted_labels)
        
        # Calculate confusion matrix values
        tp = np.sum((true_labels == 1) & (predicted_labels == 1))  # True Positive
        tn = np.sum((true_labels == 0) & (predicted_labels == 0))  # True Negative
        fp = np.sum((true_labels == 0) & (predicted_labels == 1))  # False Positive
        fn = np.sum((true_labels == 1) & (predicted_labels == 0))  # False Negative
        
        # Calculate metrics
        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
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
            'f1': f1,
            'total_samples': total
        }

    def load_merged_dataset(self):
        """
        Load the merged dataset from CSV file
        
        Returns:
            DataFrame: Loaded dataset
        """
        try:
            if not os.path.exists(self.merged_dataset_path):
                print(f"Dataset file not found: {self.merged_dataset_path}")
                return None
            
            df = pd.read_csv(self.merged_dataset_path)
            print(f"Dataset loaded: {len(df)} rows")
            print(f"Columns: {list(df.columns)}")
            
            # Ensure required columns exist
            required_columns = ['text', 'label']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"Missing required columns: {missing_columns}")
                return None
            
            return df
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None

    def evaluate_dataset(self, sample_size=None, random_seed=42):
        """
        Evaluate the model on the dataset
        
        Args:
            sample_size (int): Number of samples to evaluate (None for all)
            random_seed (int): Random seed for sampling
            
        Returns:
            tuple: (results, metrics)
        """
        # Load dataset
        df = self.load_merged_dataset()
        if df is None:
            return None, None
        
        # Sample if needed
        if sample_size and sample_size < len(df):
            print(f"Sampling {sample_size} rows from {len(df)} total rows")
            df = df.sample(n=sample_size, random_state=random_seed).reset_index(drop=True)
        
        print(f"Evaluating {len(df)} samples...")
        
        # Test model first
        print("\n=== Testing Model ===")
        test_text = "Your prompt injection is here"
        test_result = self.safe_predict(test_text)
        print(f"Test text: {test_text}")
        print(f"Test result: {test_result}")
        
        if test_result is None:
            print("Model test failed, please check model configuration")
            return None, None
        
        print(f"\n=== Model Information ===")
        print(f"Model path: {self.model_path}")
        print(f"Model config: {self.model.config}")
        if hasattr(self.model.config, 'id2label'):
            print(f"Label mapping: {self.model.config.id2label}")
        
        # Process each row
        results = []
        print(f"\n=== Processing Dataset ===")
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
            try:
                text = str(row['text'])
                
                # Skip empty text
                if not text or text.strip() == '':
                    print(f"Skipping empty text at index: {idx}")
                    continue
                
                # Get prediction
                prediction = self.safe_predict(text)
                predicted_label = self.get_predicted_label(prediction)
                
                # Record result
                result = {
                    'index': idx,
                    'text': text,
                    'true_label': row['label'],
                    'original_label': row.get('original_label'),
                    'source': row.get('source'),
                    'prediction': prediction,
                    'predicted_label': predicted_label,
                    'timestamp': datetime.now().isoformat()
                }
                
                results.append(result)
                
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                results.append({
                    'index': idx,
                    'text': str(row.get('text', 'N/A')),
                    'true_label': row.get('label'),
                    'original_label': row.get('original_label'),
                    'source': row.get('source'),
                    'prediction': None,
                    'predicted_label': None,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        # Calculate metrics
        successful_results = [r for r in results if r.get('predicted_label') is not None]
        
        if successful_results:
            true_labels = [r['true_label'] for r in successful_results]
            predicted_labels = [r['predicted_label'] for r in successful_results]
            metrics = self.calculate_metrics(true_labels, predicted_labels)
        else:
            metrics = None
            print("Warning: No successful predictions to calculate metrics")
        
        return results, metrics

    def create_output_directory(self):
        """Create output directory with model name and timestamp under results folder"""
        model_name = self.get_model_name()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join("results", f"evaluation_results_{model_name}_{timestamp}")
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        return output_dir

    def save_results(self, results, metrics, output_dir):
        """
        Save evaluation results to files
        
        Args:
            results (list): Evaluation results
            metrics (dict): Evaluation metrics
            output_dir (str): Output directory
            
        Returns:
            str: Path to summary CSV file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Prepare flattened results for CSV
        flattened_results = []
        for result in results:
            flat_result = {
                'index': result['index'],
                'text': result['text'][:200] + '...' if len(result['text']) > 200 else result['text'],
                'true_label': result.get('true_label'),
                'predicted_label': result.get('predicted_label'),
                'original_label': result.get('original_label'),
                'source': result.get('source'),
                'timestamp': result['timestamp']
            }
            
            # Add prediction comparison
            if result.get('true_label') is not None and result.get('predicted_label') is not None:
                true_label = result['true_label']
                pred_label = result['predicted_label']
                
                if true_label == 1 and pred_label == 1:
                    flat_result['prediction_result'] = 'TP'
                elif true_label == 0 and pred_label == 0:
                    flat_result['prediction_result'] = 'TN'
                elif true_label == 0 and pred_label == 1:
                    flat_result['prediction_result'] = 'FP'
                elif true_label == 1 and pred_label == 0:
                    flat_result['prediction_result'] = 'FN'
                else:
                    flat_result['prediction_result'] = 'Unknown'
            else:
                flat_result['prediction_result'] = 'N/A'
            
            # Add prediction details if available
            if result.get('prediction'):
                flat_result['predicted_label_text'] = result['prediction']['label']
                flat_result['confidence_score'] = result['prediction']['score']
            else:
                flat_result['predicted_label_text'] = None
                flat_result['confidence_score'] = None
            
            flattened_results.append(flat_result)
        
        # Save summary CSV
        summary_file = os.path.join(output_dir, f"eval_summary_{timestamp}.csv")
        df_summary = pd.DataFrame(flattened_results)
        df_summary.to_csv(summary_file, index=False, encoding='utf-8')
        print(f"Summary CSV saved to: {summary_file}")
        
        # Save detailed JSON results
        json_file = os.path.join(output_dir, f"eval_results_{timestamp}.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Detailed JSON results saved to: {json_file}")
        
        # Save metrics if available
        if metrics:
            self.save_metrics(metrics, output_dir, timestamp)
        
        return summary_file

    def save_metrics(self, metrics, output_dir, timestamp):
        """
        Save evaluation metrics to files
        
        Args:
            metrics (dict): Evaluation metrics
            output_dir (str): Output directory
            timestamp (str): Timestamp string
        """
        # Save metrics JSON
        metrics_file = os.path.join(output_dir, f"evaluation_metrics_{timestamp}.json")
        metrics_data = {
            'evaluation_time': datetime.now().isoformat(),
            'model_path': self.model_path,
            'dataset_path': self.merged_dataset_path,
            'total_samples': int(metrics.get('total_samples', 0)),  # Convert to int
            'confusion_matrix': {
                'true_positive': int(metrics['tp']),    # Convert to int
                'true_negative': int(metrics['tn']),    # Convert to int
                'false_positive': int(metrics['fp']),   # Convert to int
                'false_negative': int(metrics['fn'])    # Convert to int
            },
            'metrics': {
                'accuracy': float(metrics['accuracy']),              # Convert to float
                'recall': float(metrics['recall']),                  # Convert to float
                'precision': float(metrics['precision']),            # Convert to float
                'false_positive_rate': float(metrics['fpr']),        # Convert to float
                'f1_score': float(metrics['f1'])                     # Convert to float
            }
        }
        
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, ensure_ascii=False, indent=2)
        print(f"Metrics JSON saved to: {metrics_file}")
        
        # Save detailed text report
        report_file = os.path.join(output_dir, f"evaluation_report_{timestamp}.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("Model Evaluation Report\n")
            f.write("=" * 50 + "\n")
            f.write(f"Evaluation time: {datetime.now().isoformat()}\n")
            f.write(f"Model path: {self.model_path}\n")
            f.write(f"Dataset path: {self.merged_dataset_path}\n")
            f.write(f"Total samples: {metrics.get('total_samples', 0)}\n\n")
            
            f.write("Confusion Matrix:\n")
            f.write("-" * 20 + "\n")
            f.write(f"True Positive (TP):  {metrics['tp']}\n")
            f.write(f"True Negative (TN):  {metrics['tn']}\n")
            f.write(f"False Positive (FP): {metrics['fp']}\n")
            f.write(f"False Negative (FN): {metrics['fn']}\n\n")
            
            f.write("Evaluation Metrics:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Accuracy:    {metrics['accuracy']:.4f}\n")
            f.write(f"Recall:      {metrics['recall']:.4f}\n")
            f.write(f"Precision:   {metrics['precision']:.4f}\n")
            f.write(f"FPR:         {metrics['fpr']:.4f}\n")
            f.write(f"F1:          {metrics['f1']:.4f}\n")
    
        print(f"Evaluation report saved to: {report_file}")

    def print_summary(self, results, metrics):
        """
        Print evaluation summary
        
        Args:
            results (list): Evaluation results
            metrics (dict): Evaluation metrics
        """
        print(f"\n=== Evaluation Summary ===")
        print(f"Total samples processed: {len(results)}")
        
        successful_predictions = [r for r in results if r.get('predicted_label') is not None]
        failed_predictions = [r for r in results if r.get('predicted_label') is None]
        
        print(f"Successful predictions: {len(successful_predictions)}")
        print(f"Failed predictions: {len(failed_predictions)}")
        
        if successful_predictions:
            # Show label distribution
            label_counts = {}
            confidence_scores = []
            
            for result in successful_predictions:
                if result.get('prediction'):
                    label = result['prediction']['label']
                    score = result['prediction']['score']
                    label_counts[label] = label_counts.get(label, 0) + 1
                    confidence_scores.append(score)
            
            print(f"\nPrediction label distribution:")
            for label, count in sorted(label_counts.items()):
                print(f"  {label}: {count}")
            
            if confidence_scores:
                avg_confidence = sum(confidence_scores) / len(confidence_scores)
                print(f"\nAverage confidence: {avg_confidence:.4f}")
        
        # Print metrics if available
        if metrics:
            print(f"\n=== Evaluation Metrics ===")
            print(f"True Positive (TP): {metrics['tp']}")
            print(f"True Negative (TN): {metrics['tn']}")
            print(f"False Positive (FP): {metrics['fp']}")
            print(f"False Negative (FN): {metrics['fn']}")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"FPR: {metrics['fpr']:.4f}")
            print(f"F1: {metrics['f1']:.4f}")

def main():
    """Main function with command line argument support"""
    parser = argparse.ArgumentParser(description='Vijil Model Evaluation Script')
    parser.add_argument('--model', help='Path to vijil model directory')
    parser.add_argument('--dataset', help='Path to merged dataset CSV file')
    parser.add_argument('--sample', type=int, help='Number of samples to evaluate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = VijilModelEvaluator(args.model, args.dataset)
    
    # Create output directory
    output_dir = evaluator.create_output_directory()
    print(f"Output directory: {output_dir}")
    
    # Run evaluation
    print("Starting evaluation...")
    results, metrics = evaluator.evaluate_dataset(
        sample_size=args.sample,
        random_seed=args.seed
    )
    
    if results:
        # Save results
        summary_file = evaluator.save_results(results, metrics, output_dir)
        
        # Print summary
        evaluator.print_summary(results, metrics)
        
        print(f"\nEvaluation complete!")
        print(f"Results saved to: {output_dir}")
    else:
        print("Evaluation failed - no results generated")

if __name__ == "__main__":
    # For direct execution without command line args
    if len(os.sys.argv) == 1:
        # Default model path - modify as needed
        model_path = "./models/vijil-mbert-prompt-injection/"
        
        if not os.path.exists(model_path):
            print("Please specify model path or modify default model_path in script")
            print(f"Default path not found: {model_path}")
            exit(1)
        
        evaluator = VijilModelEvaluator(model_path)
        output_dir = evaluator.create_output_directory()
        print(f"Output directory: {output_dir}")
        
        results, metrics = evaluator.evaluate_dataset()
        
        if results:
            summary_file = evaluator.save_results(results, metrics, output_dir)
            evaluator.print_summary(results, metrics)
            print(f"\nEvaluation complete! Results saved to: {output_dir}")
        else:
            print("Evaluation failed - no results generated")
    else:
        main()