import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import argparse
import re

def validate_csv_file(file_path):
    """
    Validates that the file exists, is a CSV, and has the required headers.
    Returns the DataFrame if valid, otherwise raises appropriate exceptions.
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")
    
    # Check if file is CSV by extension
    if not file_path.lower().endswith('.csv'):
        raise ValueError(f"The file '{file_path}' is not a CSV file. Please provide a file with .csv extension.")
    
    try:
        # Read CSV file
        df = pd.read_csv(file_path)
        
        # Check for required headers (case-insensitive)
        headers = [col.lower() for col in df.columns]
        
        # Look for pair name column
        pair_col = None
        for pattern in ['pair name', 'pair_name', 'pairname', 'pair']:
            matches = [col for col in headers if pattern in col.lower()]
            if matches:
                pair_col = df.columns[headers.index(matches[0])]
                break
        
        # Look for stakeholder count column
        count_col = None
        for pattern in ['stakeholder count', 'stakeholder_count', 'stakeholdercount', 'stakeholders', 'count']:
            matches = [col for col in headers if pattern in col.lower()]
            if matches:
                count_col = df.columns[headers.index(matches[0])]
                break
        
        if pair_col is None or count_col is None:
            raise ValueError("CSV must contain 'Pair name' and 'Stakeholder count' columns (case insensitive).")
            
        # Ensure stakeholder count is numeric
        try:
            df[count_col] = pd.to_numeric(df[count_col])
        except ValueError:
            raise ValueError(f"The '{count_col}' column must contain numeric values.")
            
        return df, pair_col, count_col
        
    except pd.errors.ParserError:
        raise ValueError(f"The file '{file_path}' could not be parsed as a CSV file.")

def create_histogram(data, output_path):
    """
    Creates a histogram from the stakeholder counts and saves it to the specified path.
    """
    # Create figure and axes
    plt.figure(figsize=(10, 6))
    
    # Determine the range for bins
    min_value = int(data.min())
    max_value = int(data.max()) + 1
    bins = range(min_value, max_value + 1)
    
    # Create histogram
    sns.histplot(data, bins=bins, discrete=True, kde=False)
    
    # Customize the plot
    plt.title('Distribution of Stakeholders Identified by Pairs')
    plt.xlabel('Number of Stakeholders')
    plt.ylabel('Frequency (Number of Pairs)')
    plt.xticks(range(min_value, max_value))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of each bar
    for p in plt.gca().patches:
        height = p.get_height()
        if height > 0:
            plt.text(p.get_x() + p.get_width()/2.,
                     height + 0.1,
                     int(height),
                     ha="center")
    
    # Save the histogram
    try:
        plt.tight_layout()
        plt.savefig(output_path)
        print(f"Histogram saved successfully to '{output_path}'")
    except Exception as e:
        raise IOError(f"Failed to save histogram to '{output_path}': {str(e)}")
    
    # Close the figure to free memory
    plt.close()

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate a histogram from stakeholder count data in a CSV file.')
    parser.add_argument('csv_file', type=str, help='Path to the CSV file containing the stakeholder data.')
    parser.add_argument('output_path', type=str, help='Path where the histogram should be saved.')
    
    # Parse arguments
    args = parser.parse_args()
    
    try:
        # Validate CSV and read data
        df, pair_col, count_col = validate_csv_file(args.csv_file)
        
        # Create and save histogram
        create_histogram(df[count_col], args.output_path)
        
        # Print summary statistics
        print("\nSummary Statistics:")
        print(f"Total number of pairs: {len(df)}")
        print(f"Minimum stakeholders identified: {df[count_col].min()}")
        print(f"Maximum stakeholders identified: {df[count_col].max()}")
        print(f"Mean stakeholders identified: {df[count_col].mean():.2f}")
        print(f"Median stakeholders identified: {df[count_col].median()}")

        # Print frequency table
        value_counts = df[count_col].value_counts().sort_index()
        print("\nFrequency Distribution:")
        for value, count in value_counts.items():
            print(f"Stakeholders = {int(value)}: {count} pairs")
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()