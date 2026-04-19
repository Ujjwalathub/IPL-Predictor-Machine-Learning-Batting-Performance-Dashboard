"""
IPL Player Performance Dataset Merger
Merges all CSV files from the IPL dataset into consolidated files
"""

import pandas as pd
import os
import glob
from pathlib import Path
import re

# Configuration
DATASET_PATH = r"E:\ok\IPL - Player Performance Dataset"
OUTPUT_PATH = r"E:\ok\IPL_Merged"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Create directory if it doesn't exist
Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

def extract_year_and_metric(filename):
    """Extract year and metric type from filename"""
    # Remove .csv extension
    name = filename.replace('.csv', '')
    
    # Check for year pattern (YYYY) at the end
    year_match = re.search(r'- (\d{4})$', name)
    year = year_match.group(1) if year_match else None
    
    # Remove year from name to get metric
    metric = re.sub(r' - \d{4}$', '', name)
    metric = re.sub(r' All Seasons Combine$', '', metric)
    
    return metric, year

def merge_csv_files():
    """Main function to merge all CSV files"""
    
    # Get all CSV files
    csv_files = glob.glob(os.path.join(DATASET_PATH, "*.csv"))
    print(f"Found {len(csv_files)} CSV files\n")
    
    # Dictionary to store dataframes by metric
    metrics_data = {}
    
    # Read and categorize all files
    print("Reading CSV files...")
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        metric, year = extract_year_and_metric(filename)
        
        try:
            df = pd.read_csv(csv_file)
            
            # Add metadata columns
            df['Metric'] = metric
            if year:
                df['Year'] = int(year)
            else:
                df['Year'] = 'All'
            df['Source_File'] = filename
            
            # Store in dictionary
            if metric not in metrics_data:
                metrics_data[metric] = []
            metrics_data[metric].append(df)
            
            print(f"✓ {filename} ({len(df)} rows)")
        except Exception as e:
            print(f"✗ Error reading {filename}: {str(e)}")
    
    print(f"\nIdentified {len(metrics_data)} metrics\n")
    
    # Merge each metric group and save
    print("Creating merged files...\n")
    for metric, dfs_list in metrics_data.items():
        try:
            # Combine all dataframes for this metric
            merged_df = pd.concat(dfs_list, ignore_index=True)
            
            # Clean up column names
            merged_df.columns = merged_df.columns.str.strip()
            
            # Reorder columns - put metadata at the end
            cols = [col for col in merged_df.columns if col not in ['Metric', 'Year', 'Source_File']]
            cols.extend(['Year', 'Metric', 'Source_File'])
            merged_df = merged_df[cols]
            
            # Save to CSV
            output_file = os.path.join(OUTPUT_PATH, f"Merged_{metric}.csv")
            merged_df.to_csv(output_file, index=False)
            
            print(f"✓ {metric}")
            print(f"  - Output: Merged_{metric}.csv")
            print(f"  - Total rows: {len(merged_df)}")
            print(f"  - Columns: {len(merged_df.columns)}")
            print()
        except Exception as e:
            print(f"✗ Error merging {metric}: {str(e)}\n")
    
    # Create a master file with all data
    print("Creating master file...\n")
    try:
        all_dfs = []
        for dfs_list in metrics_data.values():
            all_dfs.extend(dfs_list)
        
        master_df = pd.concat(all_dfs, ignore_index=True)
        
        # Reorder columns
        cols = [col for col in master_df.columns if col not in ['Metric', 'Year', 'Source_File']]
        cols.extend(['Year', 'Metric', 'Source_File'])
        master_df = master_df[cols]
        
        master_file = os.path.join(OUTPUT_PATH, "IPL_Master_Dataset.csv")
        master_df.to_csv(master_file, index=False)
        
        print(f"✓ Master file created: IPL_Master_Dataset.csv")
        print(f"  - Total rows: {len(master_df)}")
        print(f"  - Total columns: {len(master_df.columns)}")
        print(f"  - File size: {os.path.getsize(master_file) / (1024*1024):.2f} MB")
        
    except Exception as e:
        print(f"✗ Error creating master file: {str(e)}")
    
    print("\n" + "="*60)
    print(f"✓ Merge completed! Files saved to: {OUTPUT_PATH}")
    print("="*60)

if __name__ == "__main__":
    merge_csv_files()
