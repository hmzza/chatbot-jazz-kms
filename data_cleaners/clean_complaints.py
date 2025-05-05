"""
clean_Complaints.py - Cleans and formats Complaint data with category-based sorting

This script processes the raw complaints_SOPs CSV file and formats Complaint entries for the chatbot,
creating separate cleaned_complaints.txt files in category-specific folders based on MainHeader.
"""

import pandas as pd
import os

# Define the categories and their folder names based on MainHeader values
CATEGORY_MAPPINGS = {
    "B2C": "B2C",
    "Digital": "Digital",
    "CFL": "CFL",
    "Jazz Rox": "jazz_rox",
    "DFS": "dfs",
    "RTG": "rtg",
    "CommUnit": "comm_unit",
    "CNC": "cnc",
    "SocialMedia": "social_media",
    "Outbound": "outbound",
    "test": "test"
}

# Map raw column names to readable labels for the chatbot
COLUMN_LABELS = {
    'C_ID': 'Complaint ID',
    'Network': 'Network Provider',
    'C_NAME': 'Complaint Name',
    'C_DESCRIPTION': 'Complaint Description',
    'Channel': 'Submission Channel',
    'c_escalation': 'Escalation Level',
    'CreateBy': 'Created By',
    'C_Type': 'Type (SOP/Complaint)',
    'Verification': 'Verification Status',
    'ComplaintPOC': 'Point of Contact',
    'FAQs': 'Frequently Asked Questions',
    'ServiceRequestActivity': 'Service Request Activity',
    'CheckList': 'Checklist Items',
    'RequiredQuestions': 'Required Questions',
    'ImportantPoints': 'Important Points',
    'ViewType': 'View Type',
    'Category': 'Category',
    'MainHeader': 'Main Header',
    'SOPStatus': 'SOP Status'
}

def create_category_directories():
    """Ensure the necessary directory structure exists for all mapped categories"""
    
    base_dir = '../cleaned_data'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    for _, folder_name in CATEGORY_MAPPINGS.items():
        category_dir = f'{base_dir}/{folder_name}'
        if not os.path.exists(category_dir):
            os.makedirs(category_dir)
    
    return base_dir

def generate_complaint_text(df_entry, main_header):
    """Generate formatted text for a specific Complaint entry"""
    sentences = []
    
    sentences.append("TYPE: COMPLAINT")
    sentences.append(f"ID: {df_entry.get('C_ID', 'Unknown ID')}")
    sentences.append("-" * 80)
    sentences.append("")  # Empty line after header

    complaint_name = df_entry.get('C_NAME', 'Unnamed Entry')
    sentence = f"📋 ENTRY: {complaint_name}\n"
    sentence += "-" * 60 + "\n"
    
    sentence += f"Main Header: {main_header}\n"

    for raw_col, label in COLUMN_LABELS.items():
        if raw_col in ['C_ID', 'C_NAME', 'MainHeader']:
            continue  # Already printed
        value = df_entry.get(raw_col, "I don't have updated information about this.")
        if pd.isna(value) or str(value).strip().lower() in ['null', '-', '']:
            value = "I don't have updated information about this."
        sentence += f"{label}: {value}\n"

    sentences.append(sentence)
    return sentences

def clean_complaints_data():
    """Clean and process the Complaint data for all categories"""
    
    try:
        # Load the CSV (try UTF-8, fallback to latin1)
        input_file = '../data/complaints_SOPs.csv'
        try:
            df = pd.read_csv(input_file, encoding='utf-8')
        except:
            df = pd.read_csv(input_file, encoding='latin1')

        print(f"📊 Loaded {len(df)} rows from {input_file}")

        # Debug: Print column names and first few rows to verify data
        print("📋 Column names:", df.columns.tolist())
        print("📋 First 5 rows sample:\n", df.head().to_string())
        print("📋 Unique C_Type values:", df['C_Type'].unique().tolist())
        print("📋 Unique MainHeader values:", df['MainHeader'].unique().tolist())

        # Remove duplicates based on C_ID
        df = df.drop_duplicates(subset=['C_ID'], keep='first')
        print(f"📊 After removing duplicates, {len(df)} rows remain")

        # Filter for Complaint entries only
        df_complaint = df[df['C_Type'].str.strip().str.lower() == 'complaint']
        print(f"📊 Found {len(df_complaint)} Complaint entries")

        # Create directories for all mapped categories
        base_dir = create_category_directories()

        # Initialize category-specific Complaint files with headers
        for main_header, folder_name in CATEGORY_MAPPINGS.items():
            complaint_file = f"{base_dir}/{folder_name}/complaints.txt"
            with open(complaint_file, "w", encoding='utf-8') as f:
                f.write(f"CATEGORY: {main_header.upper()}\n")
                f.write("TYPE: COMPLAINTS\n")
                f.write("TOTAL ENTRIES: 0 (to be updated)\n")
                f.write("-" * 80 + "\n\n")

        # Initialize general Complaint file with header
        general_complaint_file = f"{base_dir}/complaint.txt"
        with open(general_complaint_file, "w", encoding='utf-8') as f:
            f.write("CATEGORY: ALL\n")
            f.write("TYPE: COMPLAINTS\n")
            f.write("TOTAL ENTRIES: 0 (to be updated)\n")
            f.write("-" * 80 + "\n\n")

        # Track counts for updating TOTAL ENTRIES
        category_counts = {main_header: 0 for main_header in CATEGORY_MAPPINGS}
        general_count = 0

        # Process each Complaint entry
        for index, row in df_complaint.iterrows():
            main_header = str(row.get('MainHeader', 'Uncategorized')).strip()

            # Check if MainHeader is a valid category
            if main_header not in CATEGORY_MAPPINGS:
                print(f"⚠️ Skipping row {index}: MainHeader '{main_header}' not mapped to a category")
                continue

            # Generate text for the entry
            sentences = generate_complaint_text(row, main_header)

            # Update counts
            category_counts[main_header] += 1
            general_count += 1

            # Save to general Complaint file
            with open(general_complaint_file, "a", encoding='utf-8') as f:
                for line in sentences:
                    f.write(line + "\n\n")

            # Save to category-specific Complaint file
            category_folder = CATEGORY_MAPPINGS[main_header]
            category_output_file = f"{base_dir}/{category_folder}/complaints.txt"
            with open(category_output_file, "a", encoding='utf-8') as f:
                for line in sentences:
                    f.write(line + "\n\n")

            print(f"✅ Processed Complaint entry {row.get('C_ID', index)} under {main_header}")

        # Update TOTAL ENTRIES in all files
        for main_header, folder_name in CATEGORY_MAPPINGS.items():
            complaint_file = f"{base_dir}/{folder_name}/complaints.txt"
            with open(complaint_file, "r", encoding='utf-8') as f:
                content = f.readlines()
            content[2] = f"TOTAL ENTRIES: {category_counts[main_header]}\n"
            with open(complaint_file, "w", encoding='utf-8') as f:
                f.writelines(content)

        with open(general_complaint_file, "r", encoding='utf-8') as f:
            content = f.readlines()
        content[2] = f"TOTAL ENTRIES: {general_count}\n"
        with open(general_complaint_file, "w", encoding='utf-8') as f:
            f.writelines(content)

        return True

    except Exception as e:
        print(f"❌ Error processing Complaint data: {e}")
        return False

if __name__ == "__main__":
    print(f"🔍 Starting cleaning process for Complaint data...")
    clean_complaints_data()