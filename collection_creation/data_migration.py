"""
Data Migration Script for reorganizing files by category

This script helps migrate from a flat file structure to a category-based structure.
It takes the existing cleaned_data files and organizes them into category folders.
"""

import os
import shutil
import re

def create_directory_structure():
    """Create the base directory structure for categories"""
    categories = [
        "products",
        "vas",
        "jazz_rox",
        "dfs",
        "digital",
        "rtg",
        "devices"
    ]
    
    base_dir = "cleaned_data"
    
    # Create base directory if it doesn't exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # Create category directories
    for category in categories:
        category_path = os.path.join(base_dir, category)
        if not os.path.exists(category_path):
            os.makedirs(category_path)
            print(f"Created directory: {category_path}")

def analyze_current_files():
    """Analyze existing files to prepare for migration"""
    files = []
    base_dir = "cleaned_data"
    
    if os.path.exists(base_dir):
        for file in os.listdir(base_dir):
            if file.endswith(".txt"):
                file_path = os.path.join(base_dir, file)
                file_size = os.path.getsize(file_path)
                
                # Try to determine category and type from filename
                category = "products"  # Default category
                file_type = None
                
                if "offers" in file:
                    file_type = "offers"
                elif "packages" in file:
                    file_type = "packages"
                elif "data_offers" in file:
                    file_type = "data_offers"
                
                files.append({
                    "filename": file,
                    "path": file_path,
                    "size": file_size,
                    "detected_category": category,
                    "detected_type": file_type
                })
    
    return files

def migrate_files(dry_run=True):
    """
    Migrate files to category-based structure
    
    Args:
        dry_run (bool): If True, only show what would be done without making changes
    """
    create_directory_structure()
    files = analyze_current_files()
    
    print(f"{'DRY RUN: ' if dry_run else ''}Starting file migration...")
    print(f"Found {len(files)} files to process")
    
    for file_info in files:
        if not file_info["detected_type"]:
            print(f"Skipping {file_info['filename']} - could not determine file type")
            continue
        
        # For demonstration, we're using a simple mapping
        # In a real scenario, you might want to manually specify the mapping
        category_mapping = {
            "cleaned_data_offers.txt": "products",
            "cleaned_offers.txt": "products",
            "cleaned_packages.txt": "products",
            # Add mappings for other files
        }
        
        # Get category from mapping or use detected category
        category = category_mapping.get(file_info["filename"], file_info["detected_category"])
        
        # Create new filename
        new_filename = f"{file_info['detected_type']}.txt"
        new_dir = os.path.join("cleaned_data", category)
        new_path = os.path.join(new_dir, new_filename)
        
        # Check if destination already exists
        if os.path.exists(new_path):
            new_filename = f"{category}_{file_info['detected_type']}.txt"
            new_path = os.path.join(new_dir, new_filename)
        
        print(f"Moving {file_info['path']} -> {new_path}")
        
        if not dry_run:
            shutil.copy2(file_info['path'], new_path)
            print(f"Copied to new location: {new_path}")
            
            # Optional: analyze content to add category marker in the file
            update_file_content(new_path, category)

def update_file_content(file_path, category):
    """Add category markers to file content if needed"""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Check if we need to add category markers
    # This depends on your file format
    if not content.startswith(f"CATEGORY: {category}"):
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(f"CATEGORY: {category}\n\n{content}")
            print(f"Added category marker to {file_path}")

def analyze_content_for_categories():
    """
    Analyze file content to attempt to determine categories
    This is a more advanced approach for when filenames alone aren't enough
    """
    base_dir = "cleaned_data"
    results = {}
    
    category_keywords = {
        "products": ["product", "bundle", "plan"],
        "vas": ["vas", "value added", "service"],
        "jazz_rox": ["rox", "jazz rox", "music"],
        "dfs": ["dfs", "financial", "payment"],
        "digital": ["digital", "app", "online"],
        "rtg": ["rtg", "roaming"],
        "devices": ["device", "handset", "phone", "mobile"]
    }
    
    if os.path.exists(base_dir):
        for file in os.listdir(base_dir):
            if file.endswith(".txt"):
                file_path = os.path.join(base_dir, file)
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                
                category_matches = {}
                for category, keywords in category_keywords.items():
                    matches = 0
                    for keyword in keywords:
                        matches += content.count(keyword)
                    if matches > 0:
                        category_matches[category] = matches
                
                results[file] = category_matches
    
    return results

if __name__ == "__main__":
    # First run in dry_run mode to see what would happen
    # migrate_files(dry_run=True)
    
    # Uncomment to perform actual migration
    # migrate_files(dry_run=False)
    
    # For more advanced categorization based on content
    content_categories = analyze_content_for_categories()
    print("\nContent-based category detection:")
    for file, categories in content_categories.items():
        if categories:
            most_likely = max(categories.items(), key=lambda x: x[1])
            print(f"{file}: Most likely {most_likely[0]} ({most_likely[1]} matches)")
        else:
            print(f"{file}: No category matches found")