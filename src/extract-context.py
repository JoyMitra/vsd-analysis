import os
import re
import sys
import shutil

def extract_context_lines(input_dir, output_dir):
    """
    Extract lines containing the word 'context' (case insensitive) and 2 lines after
    from text files in subdirectories and place them in the same structure under 
    the output directory.
    
    Args:
        input_dir (str): The root input directory containing subdirectories
        output_dir (str): The root output directory where files will be written
    """
    # Check if input directory exists
    if not os.path.exists(input_dir):
        raise Exception(f"Input directory '{input_dir}' does not exist")
    
    # Check if output directory exists
    if os.path.exists(output_dir):
        response = input(f"Output directory '{output_dir}' already exists. Overwrite? (y/n): ")
        if response.lower() not in ['y', 'yes']:
            print("Operation cancelled")
            return
        shutil.rmtree(output_dir)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Walk through the input directory structure
    for root, dirs, files in os.walk(input_dir):
        # Get the relative path from the input directory
        rel_path = os.path.relpath(root, input_dir)
        
        # Skip the root directory itself
        if rel_path == '.':
            continue
        
        # Create corresponding output subdirectory
        output_subdir = os.path.join(output_dir, rel_path)
        os.makedirs(output_subdir, exist_ok=True)
        
        # Process all text files in the current directory
        for file in files:
            if file.endswith('.txt'):
                input_file_path = os.path.join(root, file)
                output_file_path = os.path.join(output_subdir, file)
                
                # Extract and write the context lines
                extract_and_write_context(input_file_path, output_file_path)

def extract_and_write_context(input_file, output_file):
    """
    Extract lines containing the word 'context' and 2 lines after from a text file 
    and write to output file.
    
    Args:
        input_file (str): Path to the input text file
        output_file (str): Path to the output text file
    """
    try:
        # Read the input file
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Find lines containing 'context' (case insensitive) and include 2 lines after
        context_lines = []
        pattern = re.compile(r'context', re.IGNORECASE)
        
        for i, line in enumerate(lines):
            if pattern.search(line):
                # Add the line with 'context'
                context_lines.append(line)
                
                # Add up to 2 lines after (if they exist)
                for j in range(1, 3):
                    if i + j < len(lines):
                        context_lines.append(lines[i + j])
        
        # Write the matching lines to the output file
        if context_lines:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.writelines(context_lines)
            print(f"Processed: {input_file} -> {output_file} ({len(context_lines)} matching lines)")
        else:
            print(f"No matches found in {input_file}")
            
    except Exception as e:
        print(f"Error processing {input_file}: {str(e)}")

def main():
    # Check command line arguments
    if len(sys.argv) != 3:
        print("Usage: python extract_context.py <input_directory> <output_directory>")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    try:
        extract_context_lines(input_dir, output_dir)
        print("Extraction completed successfully")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()