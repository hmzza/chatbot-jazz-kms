#!/usr/bin/env python3
"""
run_all_scripts.py - Recursively runs all Python scripts in a directory and its subdirectories

This script walks through a specified directory and all of its subdirectories,
finds all .py files, and executes them one by one using the 'python3' command.
"""

import os
import subprocess
import argparse
from datetime import datetime

def run_script(script_path):
    """Run a single Python script with python3 and print its output"""
    print("\n" + "="*80)
    print(f"📝 RUNNING: python3 {script_path}")
    print("="*80)
    
    start_time = datetime.now()
    
    # Get the directory of the script to maintain correct working directory context
    script_dir = os.path.dirname(os.path.abspath(script_path))
    script_name = os.path.basename(script_path)
    
    try:
        # Run the script using the python3 command from its own directory
        result = subprocess.run(
            ["python3", script_name],
            capture_output=True,
            text=True,
            check=False,  # Don't raise an exception if the script returns non-zero
            cwd=script_dir  # Set the working directory to the script's directory
        )
        
        # Print stdout
        if result.stdout:
            print("\n📤 STDOUT:")
            print(result.stdout)
        
        # Print stderr
        if result.stderr:
            print("\n⚠️ STDERR:")
            print(result.stderr)
        
        # Print return code
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        if result.returncode == 0:
            print(f"\n✅ Script completed successfully in {duration:.2f} seconds")
        else:
            print(f"\n❌ Script failed with return code {result.returncode} in {duration:.2f} seconds")
            
    except Exception as e:
        print(f"\n❌ Error running script: {e}")

def run_all_scripts(root_dir, exclude_dirs=None, include_patterns=None):
    """Run all Python scripts in the specified directory and its subdirectories"""
    if exclude_dirs is None:
        exclude_dirs = []
        
    if include_patterns is None:
        include_patterns = ['.py']
    
    # Convert exclude_dirs to absolute paths
    exclude_dirs = [os.path.abspath(d) for d in exclude_dirs]
    
    # Count total scripts
    total_scripts = 0
    scripts_run = 0
    scripts_skipped = 0
    scripts_failed = 0
    
    print(f"🔍 Searching for scripts in {os.path.abspath(root_dir)}...")
    
    # First, count all scripts
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Check if current directory should be excluded
        if any(os.path.abspath(dirpath).startswith(excluded) for excluded in exclude_dirs):
            continue
            
        for filename in filenames:
            if any(pattern in filename for pattern in include_patterns):
                total_scripts += 1
    
    print(f"📊 Found {total_scripts} scripts to run")
    
    # Now, run them
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Skip this directory if it's in the exclude list
        current_path = os.path.abspath(dirpath)
        if any(current_path.startswith(excluded) for excluded in exclude_dirs):
            print(f"🚫 Skipping excluded directory: {dirpath}")
            continue
        
        # Process files in this directory
        for filename in filenames:
            if any(pattern in filename for pattern in include_patterns):
                full_path = os.path.join(dirpath, filename)
                
                # Skip this script itself to avoid infinite recursion
                if os.path.samefile(full_path, os.path.abspath(__file__)):
                    print(f"⏩ Skipping self: {full_path}")
                    scripts_skipped += 1
                    continue
                
                # Run the script
                print(f"\n[{scripts_run + 1}/{total_scripts}] Running {full_path}")
                
                try:
                    run_script(full_path)
                    scripts_run += 1
                except Exception as e:
                    print(f"❌ Failed to run {full_path}: {e}")
                    scripts_failed += 1
    
    # Print summary
    print("\n" + "="*80)
    print("📋 SUMMARY")
    print("="*80)
    print(f"Total scripts found: {total_scripts}")
    print(f"Scripts run: {scripts_run}")
    print(f"Scripts skipped: {scripts_skipped}")
    print(f"Scripts failed: {scripts_failed}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all Python scripts in a directory and its subdirectories using python3")
    parser.add_argument("directory", nargs="?", default=".", help="Root directory to search for scripts (default: current directory)")
    parser.add_argument("--exclude", "-e", action="append", help="Directories to exclude (can be used multiple times)")
    parser.add_argument("--include", "-i", action="append", help="File patterns to include (default: .py)")
    
    args = parser.parse_args()
    
    run_all_scripts(args.directory, args.exclude, args.include or ['.py'])