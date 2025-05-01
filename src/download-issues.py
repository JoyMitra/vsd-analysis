import os
import json
import requests
import argparse
import sys

def get_issues_for_repo(org_repo, token):
    """
    Get all issues from a GitHub repository that have non-empty body and are not pull requests
    
    Args:
        org_repo (str): Organization and repository in format 'org/repo'
        token (str): GitHub personal access token
        
    Returns:
        list: List of issue dictionaries meeting the criteria
    """
    # GitHub API endpoint for issues
    url = f"https://api.github.com/repos/{org_repo}/issues"
    
    # Set request headers
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    
    # Parameters for the request
    params = {
        "state": "all",   # Get both open and closed issues
        "per_page": 100,  # Maximum items per page
        "page": 1         # Start with first page
    }
    
    all_issues = []
    
    while True:
        # Make the request
        response = requests.get(url, headers=headers, params=params)
        
        # Check if request was successful
        if response.status_code != 200:
            print(f"Error retrieving issues from {org_repo}: {response.status_code}")
            print(f"Response: {response.text}")
            return []
        
        # Parse response JSON
        issues_page = response.json()
        
        # If no issues are returned, we've reached the end
        if not issues_page:
            break
            
        # Filter issues: keep only those with non-empty body and no pull_request property
        for issue in issues_page:
            # Check if it's not a pull request and has a non-empty body
            if 'pull_request' not in issue and issue.get('body') and issue['body'].strip():
                all_issues.append(issue)
        
        # Check if there are more pages
        if 'Link' in response.headers and 'rel="next"' in response.headers['Link']:
            params['page'] += 1
        else:
            break
    
    return all_issues

def sanitize_filename(s):
    """
    Sanitize a string to be used as a filename by removing invalid characters
    """
    illegal_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for char in illegal_chars:
        s = s.replace(char, '_')
    return s

def save_issues_to_files(issues, repo_name, output_dir):
    """
    Save issues to text files in a repository-specific directory
    
    Args:
        issues (list): List of issue dictionaries
        repo_name (str): Repository name
        output_dir (str): Base output directory
    """
    # Create directory for this repository
    repo_dir = os.path.join(output_dir, sanitize_filename(repo_name))
    os.makedirs(repo_dir, exist_ok=True)
    
    # Process each issue
    for issue in issues:
        # Create filename from issue number and title
        filename = f"issue_{issue['number']}_{sanitize_filename(issue['title'])}.txt"
        filepath = os.path.join(repo_dir, filename)
        
        # Prepare issue content
        content = f"Issue #{issue['number']}: {issue['title']}\n"
        content += f"State: {issue['state']}\n"
        content += f"Created by: {issue['user']['login']}\n"
        content += f"Created at: {issue['created_at']}\n"
        
        if issue.get('closed_at'):
            content += f"Closed at: {issue['closed_at']}\n"
            
        content += f"URL: {issue['html_url']}\n\n"
        content += f"Description:\n{issue['body']}\n"
        
        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
            
        print(f"Saved issue #{issue['number']} to {filepath}")

def main():
    parser = argparse.ArgumentParser(description='Download GitHub issues to text files')
    parser.add_argument('--input', required=True, help='Path to JSON file containing org/repo entries')
    parser.add_argument('--output', required=True, help='Path to output directory')
    parser.add_argument('--token', required=True, help='GitHub personal access token')
    
    args = parser.parse_args()
    
    # Check if output directory exists
    if os.path.exists(args.output):
        response = input(f"Output directory '{args.output}' already exists. Override? (yes/no): ")
        if response.lower() != 'yes':
            print("Operation cancelled.")
            sys.exit(0)
    else:
        os.makedirs(args.output, exist_ok=True)
    
    # Read input JSON file
    try:
        with open(args.input, 'r') as f:
            repos = json.load(f)
    except Exception as e:
        print(f"Error reading input file: {str(e)}")
        sys.exit(1)
    
    # Process each repo
    for repo_entry in repos:
        print(f"Processing repository: {repo_entry}")
        
        # Get all issues for this repo
        issues = get_issues_for_repo(repo_entry, args.token)
        print(f"Found {len(issues)} issues with non-empty body that are not pull requests")
        
        # Save issues to files
        if issues:
            save_issues_to_files(issues, repo_entry.split('/')[-1], args.output)
    
    print("All issues have been processed and saved.")

if __name__ == '__main__':
    main()