import os
import pandas as pd
import json
import glob
from pathlib import Path
from datetime import datetime
import logging
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetMerger:
    def __init__(self, datasets_dir="dataset-source", output_file="merged_datasets.csv"):
        self.datasets_dir = Path(datasets_dir)
        self.output_file = output_file
        self.supported_formats = ['.csv', '.tsv', '.json', '.parquet']
        
        # Define threat label patterns (case-insensitive)
        self.threat_patterns = [
            'jailbreak', 'direct', 'indirect', 'injection', 'attack', 'malicious',
            'harmful', 'adversarial', 'security-violating', 'threat', 'toxic',
            'unsafe', 'violation', '1'  # Add number 1 as threat label
        ]
        
        # Define safe label patterns (case-insensitive)
        self.safe_patterns = [
            'benign', 'safe', 'normal', 'legitimate', 'clean', 'harmless',
            'adversarial_benign', '0'  # Add number 0 as safe label
        ]
    
    def classify_threat_level(self, original_label):
        """
        Classify threat level based on original label
        Returns: 1 (threat), 0 (safe)
        """
        if pd.isna(original_label) or original_label == "":
            return 0  # Default to safe
        
        label_str = str(original_label).lower().strip()
        
        # First check for explicit threat labels
        for pattern in self.threat_patterns:
            if pattern in label_str:
                return 1
        
        # Then check for explicit safe labels
        for pattern in self.safe_patterns:
            if pattern in label_str:
                return 0
        
        # If no match found, log warning and default to safe
        logger.warning(f"Unknown label pattern: {original_label}, defaulting to 0 (safe)")
        return 0
        
    def standardize_columns(self, df, source_file):
        """
        Standardize column names, keep only necessary four columns
        """
        standardized_df = pd.DataFrame()
        
        # Extract source name from file path
        source_name = Path(source_file).stem
        
        # Common text column names
        text_columns = ['text', 'prompt', 'user_input', 'adversarial', 'question']
        # Common label column names
        label_columns = ['label', 'type', 'injection_type', 'risk_category']
        
        # Find text column
        text_col = None
        for col in df.columns:
            if col.lower() in [tc.lower() for tc in text_columns]:
                text_col = col
                break
        
        # If no standard text column found, use first string column
        if text_col is None:
            for col in df.columns:
                if df[col].dtype == 'object':
                    text_col = col
                    break
        
        # Find label column
        label_col = None
        for col in df.columns:
            if col.lower() in [lc.lower() for lc in label_columns]:
                label_col = col
                break
        
        # If no label column found, use second column (if exists)
        if label_col is None and len(df.columns) > 1:
            for col in df.columns:
                if col != text_col:
                    label_col = col
                    break
        
        # Build standardized dataframe - keep four columns
        if text_col and not df[text_col].empty:
            # Save original labels
            if label_col:
                original_labels = df[label_col].astype(str)
            else:
                original_labels = "unknown"
                logger.warning(f"No label column found in {source_file}")
            
            # Create binary labels
            binary_labels = [self.classify_threat_level(label) for label in original_labels]
            
            # Create only the needed four columns
            standardized_df['text'] = df[text_col].astype(str)
            standardized_df['label'] = binary_labels
            standardized_df['source'] = source_name
            standardized_df['original_label'] = original_labels
            
        else:
            logger.warning(f"No valid text column found in {source_file}")
            return pd.DataFrame()  # Return empty dataframe
        
        return standardized_df
    
    def read_csv_file(self, file_path):
        """Read CSV file"""
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
            logger.info(f"Successfully read CSV file: {file_path} ({len(df)} rows)")
            return df
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(file_path, encoding='latin-1')
                logger.info(f"Successfully read CSV file with latin-1 encoding: {file_path} ({len(df)} rows)")
                return df
            except Exception as e:
                logger.error(f"Error reading CSV file {file_path}: {e}")
                return None
        except Exception as e:
            logger.error(f"Error reading CSV file {file_path}: {e}")
            return None
    
    def read_tsv_file(self, file_path):
        """Read TSV file"""
        try:
            df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
            logger.info(f"Successfully read TSV file: {file_path} ({len(df)} rows)")
            return df
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(file_path, sep='\t', encoding='latin-1')
                logger.info(f"Successfully read TSV file with latin-1 encoding: {file_path} ({len(df)} rows)")
                return df
            except Exception as e:
                logger.error(f"Error reading TSV file {file_path}: {e}")
                return None
        except Exception as e:
            logger.error(f"Error reading TSV file {file_path}: {e}")
            return None
    
    def read_json_file(self, file_path):
        """Read JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # If it's a list format
            if isinstance(data, list):
                df = pd.DataFrame(data)
            # If it's a dictionary format, try to convert
            elif isinstance(data, dict):
                df = pd.DataFrame([data])
            else:
                logger.error(f"Unsupported JSON format in {file_path}")
                return None
                
            logger.info(f"Successfully read JSON file: {file_path} ({len(df)} rows)")
            return df
        except Exception as e:
            logger.error(f"Error reading JSON file {file_path}: {e}")
            return None
    
    def read_parquet_file(self, file_path):
        """Read Parquet file"""
        try:
            df = pd.read_parquet(file_path)
            logger.info(f"Successfully read Parquet file: {file_path} ({len(df)} rows)")
            return df
        except Exception as e:
            logger.error(f"Error reading Parquet file {file_path}: {e}")
            return None
    
    def find_all_dataset_files(self):
        """Find all dataset files with supported formats"""
        all_files = []
        
        for format_ext in self.supported_formats:
            # Search for files in current directory and subdirectories
            pattern = str(self.datasets_dir / f"**/*{format_ext}")
            files = glob.glob(pattern, recursive=True)
            all_files.extend(files)
        
        logger.info(f"Found {len(all_files)} dataset files: {[Path(f).name for f in all_files]}")
        return all_files
    
    def merge_datasets(self):
        """Merge all datasets"""
        all_files = self.find_all_dataset_files()
        
        if not all_files:
            logger.warning("No dataset files found!")
            return
        
        merged_data = []
        label_mapping_stats = {}
        
        for file_path in all_files:
            file_ext = Path(file_path).suffix.lower()
            
            # Choose reading method based on file format
            if file_ext == '.csv':
                df = self.read_csv_file(file_path)
            elif file_ext == '.tsv':
                df = self.read_tsv_file(file_path)
            elif file_ext == '.json':
                df = self.read_json_file(file_path)
            elif file_ext == '.parquet':
                df = self.read_parquet_file(file_path)
            else:
                logger.warning(f"Unsupported file format: {file_path}")
                continue
            
            if df is not None and not df.empty:
                # Standardize column names and format
                standardized_df = self.standardize_columns(df, file_path)
                if not standardized_df.empty:
                    merged_data.append(standardized_df)
                    
                    # Collect label mapping statistics
                    source_name = Path(file_path).stem
                    original_labels = standardized_df['original_label'].value_counts()
                    binary_labels = standardized_df['label'].value_counts()
                    label_mapping_stats[source_name] = {
                        'original_labels': dict(original_labels),
                        'binary_labels': dict(binary_labels),
                        'total_rows': len(standardized_df)
                    }
                    
                    logger.info(f"Added {len(standardized_df)} rows from {Path(file_path).name}")
                else:
                    logger.warning(f"No valid data extracted from: {file_path}")
            else:
                logger.warning(f"Skipped empty or invalid file: {file_path}")
        
        if merged_data:
            # Merge all data
            final_df = pd.concat(merged_data, ignore_index=True)
            
            # Remove duplicates (based on text and original_label)
            initial_count = len(final_df)
            final_df = final_df.drop_duplicates(subset=['text', 'original_label'], keep='first')
            final_count = len(final_df)
            
            if initial_count != final_count:
                logger.info(f"Removed {initial_count - final_count} duplicate rows")
            
            # Save merged data
            final_df.to_csv(self.output_file, index=False, encoding='utf-8')
            
            logger.info(f"Successfully merged {len(merged_data)} files into {self.output_file}")
            logger.info(f"Total rows: {len(final_df)}")
            logger.info(f"Columns: {list(final_df.columns)}")
            
            # Display statistics
            self.print_statistics(final_df, label_mapping_stats)
            
        else:
            logger.error("No valid data found to merge!")
    
    def print_statistics(self, df, label_mapping_stats):
        """Print data statistics"""
        print("\n" + "="*60)
        print("DATASET MERGE STATISTICS")
        print("="*60)
        print(f"Total rows: {len(df):,}")
        print(f"Columns: {list(df.columns)}")
        
        # Overall label distribution
        print(f"\nOverall Binary Label Distribution:")
        label_counts = df['label'].value_counts().sort_index()
        total = len(df)
        for label, count in label_counts.items():
            percentage = (count / total) * 100
            threat_type = "Threat" if label == 1 else "Safe"
            print(f"  {label} ({threat_type}): {count:,} rows ({percentage:.1f}%)")
        
        # Statistics by source
        print(f"\nRows by Source:")
        source_counts = df['source'].value_counts()
        for source, count in source_counts.items():
            print(f"  {source}: {count:,} rows")
        
        # Detailed label mapping statistics
        print(f"\nLabel Mapping Details:")
        for source, stats in label_mapping_stats.items():
            print(f"\n  {source} ({stats['total_rows']:,} rows):")
            print(f"    Original labels: {stats['original_labels']}")
            print(f"    Binary labels: {stats['binary_labels']}")
        
        # Show complete distribution of original labels
        print(f"\nOriginal Label Distribution (Top 20):")
        original_label_counts = df['original_label'].value_counts().head(20)
        for label, count in original_label_counts.items():
            percentage = (count / total) * 100
            print(f"  '{label}': {count:,} rows ({percentage:.1f}%)")
        
        # Show sample data
        print(f"\nSample data (first 3 rows):")
        print(df[['text', 'label', 'source', 'original_label']].head(3).to_string(index=False, max_colwidth=50))
        print("="*60)

def main():
    """Main function"""
    # Create output filename
    output_file = f"merged_datasets.csv"
    
    # Create merger instance
    merger = DatasetMerger(output_file=output_file)
    
    # Execute merge
    print("Starting dataset merge process...")
    print("Converting labels to binary format: 1=Threat, 0=Safe")
    merger.merge_datasets()
    print(f"\nMerge complete! Output saved to: {output_file}")

if __name__ == "__main__":
    main()