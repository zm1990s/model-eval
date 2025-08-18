"""
Create a balanced short dataset from merged_datasets.csv
This script samples specified number of samples with balanced sources and labels.
"""

import pandas as pd
import numpy as np
from collections import Counter
import os
import argparse

def create_short_dataset(input_file="merged_datasets.csv", 
                        output_file=None,
                        target_samples=1000,
                        random_seed=42):
    """
    Create a balanced short dataset from the full merged dataset
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file (auto-generated if None)
        target_samples (int): Target number of samples
        random_seed (int): Random seed for reproducibility
    """
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Load the full dataset
    print(f"Loading dataset from: {input_file}")
    try:
        df = pd.read_csv(input_file, encoding='utf-8')
        print(f"Loaded dataset with {len(df)} rows")
        print(f"Columns: {list(df.columns)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None
    
    # Display original distribution
    print(f"\n=== Original Dataset Distribution ===")
    
    # Label distribution
    if 'label' in df.columns:
        label_counts = df['label'].value_counts().sort_index()
        print(f"Label distribution:")
        for label, count in label_counts.items():
            percentage = (count / len(df)) * 100
            threat_type = "Threat" if label == 1 else "Safe"
            print(f"  {label} ({threat_type}): {count:,} ({percentage:.1f}%)")
    
    # Source distribution
    if 'source' in df.columns:
        source_counts = df['source'].value_counts()
        print(f"\nSource distribution:")
        for source, count in source_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {source}: {count:,} ({percentage:.1f}%)")
    
    # Calculate samples per source
    unique_sources = df['source'].unique()
    num_sources = len(unique_sources)
    base_samples_per_source = target_samples // num_sources
    remainder = target_samples % num_sources
    
    print(f"\n=== Sampling Strategy ===")
    print(f"Target samples: {target_samples}")
    print(f"Total sources: {num_sources}")
    print(f"Base samples per source: {base_samples_per_source}")
    print(f"Remainder to distribute: {remainder}")
    
    # Create sampling plan
    sampling_plan = {}
    for i, source in enumerate(unique_sources):
        # Distribute remainder among first few sources
        extra = 1 if i < remainder else 0
        samples_for_source = base_samples_per_source + extra
        sampling_plan[source] = samples_for_source
        print(f"  {source}: {samples_for_source} samples")
    
    # Sample from each source with label balance
    sampled_data = []
    
    print(f"\n=== Sampling Process ===")
    for source, target_count in sampling_plan.items():
        source_data = df[df['source'] == source].copy()
        
        if len(source_data) == 0:
            print(f"Warning: No data found for source {source}")
            continue
        
        print(f"\nProcessing {source}:")
        print(f"  Available samples: {len(source_data)}")
        print(f"  Target samples: {target_count}")
        
        # Check label distribution in this source
        source_label_counts = source_data['label'].value_counts().sort_index()
        print(f"  Label distribution: {dict(source_label_counts)}")
        
        if len(source_data) <= target_count:
            # Use all available data if we have fewer samples than needed
            sampled_source_data = source_data
            print(f"  Using all {len(sampled_source_data)} available samples")
        else:
            # Try to maintain label balance within each source
            label_0_data = source_data[source_data['label'] == 0]
            label_1_data = source_data[source_data['label'] == 1]
            
            # Calculate target for each label (try to maintain proportion)
            total_label_0 = len(label_0_data)
            total_label_1 = len(label_1_data)
            total_available = total_label_0 + total_label_1
            
            if total_available > 0:
                # Proportional sampling
                target_label_0 = int(target_count * total_label_0 / total_available)
                target_label_1 = target_count - target_label_0
                
                # Adjust if we don't have enough samples for a label
                actual_label_0 = min(target_label_0, total_label_0)
                actual_label_1 = min(target_label_1, total_label_1)
                
                # If we still have quota, redistribute
                remaining_quota = target_count - actual_label_0 - actual_label_1
                if remaining_quota > 0:
                    if total_label_0 > actual_label_0:
                        add_to_0 = min(remaining_quota, total_label_0 - actual_label_0)
                        actual_label_0 += add_to_0
                        remaining_quota -= add_to_0
                    if remaining_quota > 0 and total_label_1 > actual_label_1:
                        add_to_1 = min(remaining_quota, total_label_1 - actual_label_1)
                        actual_label_1 += add_to_1
                
                print(f"  Sampling: {actual_label_0} safe + {actual_label_1} threat = {actual_label_0 + actual_label_1}")
                
                # Sample from each label
                sampled_parts = []
                if actual_label_0 > 0 and len(label_0_data) > 0:
                    sampled_0 = label_0_data.sample(n=actual_label_0, random_state=random_seed)
                    sampled_parts.append(sampled_0)
                
                if actual_label_1 > 0 and len(label_1_data) > 0:
                    sampled_1 = label_1_data.sample(n=actual_label_1, random_state=random_seed)
                    sampled_parts.append(sampled_1)
                
                if sampled_parts:
                    sampled_source_data = pd.concat(sampled_parts, ignore_index=True)
                else:
                    sampled_source_data = pd.DataFrame()
            else:
                sampled_source_data = pd.DataFrame()
        
        if not sampled_source_data.empty:
            sampled_data.append(sampled_source_data)
            print(f"  Actually sampled: {len(sampled_source_data)} samples")
    
    # Combine all sampled data
    if sampled_data:
        final_df = pd.concat(sampled_data, ignore_index=True)
        
        # Shuffle the final dataset
        final_df = final_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
        
        actual_samples = len(final_df)
        
        # Auto-generate output filename based on actual sample count
        if output_file is None:
            base_name = os.path.splitext(input_file)[0]
            output_file = f"{base_name}_short_{actual_samples}.csv"
        
        print(f"\n=== Final Short Dataset ===")
        print(f"Target samples: {target_samples}")
        print(f"Actual samples: {actual_samples}")
        
        # Display final distributions
        if 'label' in final_df.columns:
            final_label_counts = final_df['label'].value_counts().sort_index()
            print(f"Final label distribution:")
            for label, count in final_label_counts.items():
                percentage = (count / len(final_df)) * 100
                threat_type = "Threat" if label == 1 else "Safe"
                print(f"  {label} ({threat_type}): {count:,} ({percentage:.1f}%)")
        
        if 'source' in final_df.columns:
            final_source_counts = final_df['source'].value_counts()
            print(f"Final source distribution:")
            for source, count in final_source_counts.items():
                percentage = (count / len(final_df)) * 100
                print(f"  {source}: {count:,} ({percentage:.1f}%)")
        
        # Save the short dataset
        final_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\nShort dataset saved to: {output_file}")
        
        # Display sample data
        print(f"\nSample data (first 3 rows):")
        print(final_df[['text', 'label', 'source', 'original_label']].head(3).to_string(index=False, max_colwidth=50))
        
        return final_df, actual_samples
    else:
        print("Error: No data was sampled")
        return None, 0

def main():
    """Main function with command line argument support"""
    parser = argparse.ArgumentParser(description='Create balanced short dataset from merged_datasets.csv')
    parser.add_argument('--input', default='merged_datasets.csv', 
                       help='Input CSV file path (default: merged_datasets.csv)')
    parser.add_argument('--output', 
                       help='Output CSV file path (auto-generated if not specified)')
    parser.add_argument('--samples', type=int, default=1000,
                       help='Number of samples to extract (default: 1000)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    print(f"Creating balanced short dataset with target of {args.samples} samples...")
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        print("Please ensure the merged dataset file exists in the current directory.")
        return
    
    # Create short dataset
    short_df, actual_samples = create_short_dataset(
        input_file=args.input,
        output_file=args.output,
        target_samples=args.samples,
        random_seed=args.seed
    )
    
    if short_df is not None:
        if args.output:
            output_name = args.output
        else:
            base_name = os.path.splitext(args.input)[0]
            output_name = f"{base_name}_short_{actual_samples}.csv"
        
        print(f"\nSuccess! Created balanced short dataset with {actual_samples} samples.")
        print(f"File saved as: {output_name}")
    else:
        print("Failed to create short dataset.")

if __name__ == "__main__":
    # For direct execution without command line args
    if len(os.sys.argv) == 1:
        print("Creating default short dataset (target: 1000 samples)...")
        
        input_file = "merged_datasets.csv"
        
        if not os.path.exists(input_file):
            print(f"Error: Input file not found: {input_file}")
            print("Please run merge_datasets.py first to create the merged dataset.")
            exit(1)
        
        short_df, actual_samples = create_short_dataset(
            input_file=input_file,
            target_samples=1000,
            random_seed=42
        )
        
        if short_df is not None:
            print(f"\nSuccess! Created balanced short dataset with {actual_samples} samples.")
            print(f"File saved as: merged_datasets_short_{actual_samples}.csv")
        else:
            print("Failed to create short dataset.")
    else:
        main()