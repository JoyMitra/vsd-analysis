import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import sys
import argparse

def validate_csv_file(file_path, group_name):
    """
    Validates that the file exists and extracts stakeholder count data.
    Returns the DataFrame if valid, otherwise raises appropriate exceptions.
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The {group_name} file '{file_path}' does not exist.")
    
    try:
        # Read CSV file
        df = pd.read_csv(file_path)
        
        # Look for pair name column
        pair_col = None
        count_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            # Find pair name column
            if any(term in col_lower for term in ['pair', 'team', 'group']):
                pair_col = col
            # Find stakeholder count column
            elif any(term in col_lower for term in ['stakeholder', 'count', 'number']):
                count_col = col
        
        # If columns weren't found by keyword, use positional assumption
        if pair_col is None and count_col is None and len(df.columns) >= 2:
            pair_col = df.columns[0]
            count_col = df.columns[1]
        
        if pair_col is None or count_col is None:
            raise ValueError(f"Could not identify pair name and stakeholder count columns in {group_name} file.")
            
        # Ensure stakeholder count is numeric
        try:
            df[count_col] = pd.to_numeric(df[count_col])
        except ValueError:
            raise ValueError(f"The stakeholder count column in {group_name} file must contain numeric values.")
            
        return df, pair_col, count_col
        
    except pd.errors.ParserError:
        raise ValueError(f"The file '{file_path}' could not be parsed as a CSV file.")

def perform_mann_whitney_test(group1_data, group2_data, group1_name, group2_name, alpha):
    """
    Performs the Mann-Whitney U test and returns the results.
    """
    # Perform the Mann-Whitney U test
    u_stat, p_value = stats.mannwhitneyu(
        group1_data, 
        group2_data,
        alternative='two-sided'  # Testing for any difference between groups
    )
    
    # Calculate additional metrics
    median1 = np.median(group1_data)
    median2 = np.median(group2_data)
    mean1 = np.mean(group1_data)
    mean2 = np.mean(group2_data)
    
    # Calculate effect size (Common Language Effect Size)
    def cles(x, y):
        """Compute Common Language Effect Size."""
        count = 0
        for i in x:
            for j in y:
                if i > j:
                    count += 1
                elif i == j:
                    count += 0.5
        return count / (len(x) * len(y))
    
    effect_size = cles(group1_data, group2_data)
    
    # Calculate rank-biserial correlation (another effect size measure)
    n1, n2 = len(group1_data), len(group2_data)
    rank_biserial = - 1 + (2 * u_stat) / (n1 * n2)
    
    # Return results
    results = {
        'u_statistic': u_stat,
        'p_value': p_value,
        'significant': p_value < alpha,
        'alpha': alpha,
        'group1_median': median1,
        'group2_median': median2,
        'group1_mean': mean1,
        'group2_mean': mean2,
        'effect_size': effect_size,
        'rank_biserial': rank_biserial,
        'group1_name': group1_name,
        'group2_name': group2_name
    }
    
    return results

def create_comparison_plots(group1_data, group2_data, group1_name, group2_name, alpha, results):
    """
    Creates visualizations to compare the two groups.
    """
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Box plot
    box_data = [group1_data, group2_data]
    ax1.boxplot(box_data, labels=[group1_name, group2_name])
    ax1.set_title('Box Plot of Stakeholders by Group')
    ax1.set_ylabel('Number of Stakeholders')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add data points jittered for better visualization
    for i, data in enumerate([group1_data, group2_data]):
        x = np.random.normal(i+1, 0.04, size=len(data))
        ax1.plot(x, data, 'o', alpha=0.5, zorder=1)
    
    # Add p-value annotation to the box plot
    p_text = f"p = {results['p_value']:.4f}"
    if results['significant']:
        p_text += f" < {alpha} (significant)"
    else:
        p_text += f" > {alpha} (not significant)"
    ax1.annotate(p_text, xy=(0.5, 0.01), xycoords='axes fraction',
               ha='center', va='center', fontsize=10,
               bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3))
    
    # Histogram
    max_val = max(max(group1_data), max(group2_data))
    min_val = min(min(group1_data), min(group2_data))
    bins = range(min_val, max_val + 2)  # +2 to include the max value
    
    ax2.hist([group1_data, group2_data], bins=bins, alpha=0.7, 
             label=[group1_name, group2_name], edgecolor='black')
    ax2.set_title('Histogram of Stakeholders by Group')
    ax2.set_xlabel('Number of Stakeholders')
    ax2.set_ylabel('Frequency')
    ax2.set_xticks(range(min_val, max_val + 1))
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add effect size annotation to the histogram
    effect_text = f"Effect size: {results['effect_size']:.2f}"
    if results['effect_size'] > 0.71:
        effect_text += " (large)"
    elif results['effect_size'] > 0.64:
        effect_text += " (medium)"
    else:
        effect_text += " (small)"
    ax2.annotate(effect_text, xy=(0.5, 0.01), xycoords='axes fraction',
               ha='center', va='center', fontsize=10,
               bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3))
    
    # Add an overall title with the conclusion
    if results['significant']:
        if results['group1_median'] > results['group2_median']:
            conclusion = f"{group1_name} identified significantly more stakeholders than {group2_name}"
        else:
            conclusion = f"{group2_name} identified significantly more stakeholders than {group1_name}"
    else:
        conclusion = "No significant difference in stakeholders identified between groups"
    
    plt.suptitle(conclusion, fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for suptitle
    
    return fig

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Perform Mann-Whitney U test on stakeholder count data from two CSV files.')
    parser.add_argument('group1_file', type=str, help='Path to the first CSV file (Group 1).')
    parser.add_argument('group2_file', type=str, help='Path to the second CSV file (Group 2).')
    parser.add_argument('--group1_name', type=str, default='Group 1', help='Name for the first group (default: Group 1).')
    parser.add_argument('--group2_name', type=str, default='Group 2', help='Name for the second group (default: Group 2).')
    parser.add_argument('--plot', type=str, help='Path to save the comparison plots (optional).')
    parser.add_argument('--alpha', type=float, default=0.05, choices=[0.05, 0.01], 
                        help='Significance level (alpha) for hypothesis testing (default: 0.05). Choose 0.05 or 0.01.')
    parser.add_argument('--alternative', type=str, default='two-sided', choices=['two-sided', 'greater', 'less'],
                        help='Alternative hypothesis. Default is "two-sided" (any difference). Use "greater" if Group 1 > Group 2, or "less" if Group 1 < Group 2.')
    
    # Parse arguments
    args = parser.parse_args()
    
    try:
        # Validate CSV files and read data
        group1_df, group1_pair_col, group1_count_col = validate_csv_file(args.group1_file, args.group1_name)
        group2_df, group2_pair_col, group2_count_col = validate_csv_file(args.group2_file, args.group2_name)
        
        # Extract stakeholder counts
        group1_counts = group1_df[group1_count_col].values
        group2_counts = group2_df[group2_count_col].values
        
        # Print basic information about each group
        print(f"\n{args.group1_name} Statistics:")
        print(f"Number of pairs: {len(group1_counts)}")
        print(f"Mean stakeholders: {group1_counts.mean():.2f}")
        print(f"Median stakeholders: {np.median(group1_counts):.1f}")
        print(f"Range: {group1_counts.min()} to {group1_counts.max()}")
        print(f"Value counts: {pd.Series(group1_counts).value_counts().sort_index().to_dict()}")
        
        print(f"\n{args.group2_name} Statistics:")
        print(f"Number of pairs: {len(group2_counts)}")
        print(f"Mean stakeholders: {group2_counts.mean():.2f}")
        print(f"Median stakeholders: {np.median(group2_counts):.1f}")
        print(f"Range: {group2_counts.min()} to {group2_counts.max()}")
        print(f"Value counts: {pd.Series(group2_counts).value_counts().sort_index().to_dict()}")
        
        # Check for normality (Shapiro-Wilk test)
        print("\nNormality Tests (Shapiro-Wilk):")
        _, p_norm1 = stats.shapiro(group1_counts)
        _, p_norm2 = stats.shapiro(group2_counts)
        
        print(f"{args.group1_name}: p = {p_norm1:.4f} ({'Normal' if p_norm1 >= 0.05 else 'Not normal'})")
        print(f"{args.group2_name}: p = {p_norm2:.4f} ({'Normal' if p_norm2 >= 0.05 else 'Not normal'})")
        
        if p_norm1 < 0.05 or p_norm2 < 0.05:
            print("At least one group is not normally distributed, confirming Mann-Whitney U is appropriate.")
        
        # Perform Mann-Whitney U test
        results = perform_mann_whitney_test(
            group1_counts, 
            group2_counts, 
            args.group1_name, 
            args.group2_name,
            args.alpha
        )
        
        # Print results
        print(f"\nMann-Whitney U Test Results (α = {args.alpha}):")
        print(f"U statistic: {results['u_statistic']:.1f}")
        print(f"p-value: {results['p_value']:.4f}")
        
        # Interpret the results
        if results['significant']:
            print(f"Conclusion: There is a significant difference in the number of stakeholders identified between the two groups (p < {args.alpha}).")
            if results['group1_median'] > results['group2_median']:
                print(f"  {args.group1_name} identified significantly more stakeholders than {args.group2_name}.")
            else:
                print(f"  {args.group2_name} identified significantly more stakeholders than {args.group1_name}.")
        else:
            print(f"Conclusion: There is no significant difference in the number of stakeholders identified between the two groups (p >= {args.alpha}).")
        
        print(f"Effect size (CLES): {results['effect_size']:.2f}")
        if results['effect_size'] > 0.71:
            print("  This represents a large effect size.")
        elif results['effect_size'] > 0.64:
            print("  This represents a medium effect size.")
        else:
            print("  This represents a small effect size.")
        
        print(f"Rank-biserial correlation: {results['rank_biserial']:.2f}")
        
        # Create and save plots if requested
        if args.plot:
            fig = create_comparison_plots(
                group1_counts, 
                group2_counts,
                args.group1_name, 
                args.group2_name,
                args.alpha,
                results
            )
            fig.savefig(args.plot, dpi=300, bbox_inches='tight')
            print(f"\nComparison plots saved to '{args.plot}'")
            
            # Create additional plot: ECDF (Empirical Cumulative Distribution Function)
            plt.figure(figsize=(10, 6))
            
            # Calculate ECDF for both groups
            def ecdf(data):
                x = np.sort(data)
                y = np.arange(1, len(data) + 1) / len(data)
                return x, y
            
            x1, y1 = ecdf(group1_counts)
            x2, y2 = ecdf(group2_counts)
            
            plt.step(x1, y1, where='post', label=args.group1_name)
            plt.step(x2, y2, where='post', label=args.group2_name)
            
            plt.xlabel('Number of Stakeholders')
            plt.ylabel('Cumulative Probability')
            plt.title('Empirical Cumulative Distribution Function (ECDF)')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Save ECDF plot
            ecdf_path = os.path.splitext(args.plot)[0] + '_ecdf' + os.path.splitext(args.plot)[1]
            plt.savefig(ecdf_path, dpi=300, bbox_inches='tight')
            print(f"ECDF plot saved to '{ecdf_path}'")
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()