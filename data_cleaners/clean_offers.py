"""
clean_offers.py - Cleans and formats offers data with category-based sorting

This script processes the raw offers CSV file and formats it for the chatbot,
creating separate output files based on the MainHeader category.
"""

import pandas as pd
import os

# Define the categories and their folder names
CATEGORY_MAPPINGS = {
    "B2C": "B2C",
    "Digital": "Digital", 
    "Jazz Rox": "jazz_rox",
    "CFL": "CFL"
}

# Map raw column names to readable labels
COLUMN_LABELS = {
    'OFFER_NAME': 'Offer Name',
    'INCENTIVES': 'Incentives',
    'VALIDITY': 'Validity',
    'TIME_DURATION': 'Time Duration',
    'RECURSIVE': 'Is Recursive',
    'DEPOSIT': 'Deposit Info',
    'SUB': 'Subscription Code',
    'UNSUB': 'Unsubscription Code',
    'INFO': 'Information Code',
    'OFFER_CAT': 'Offer Category',
    'CHARGES': 'Charges',
    'OFFERTYPE': 'Offer Type',
    'MAINHEADER': 'Main Header'
}

def create_category_directories():
    """Ensure the necessary directory structure exists for all categories"""
    
    # Create base directory if it doesn't exist
    if not os.path.exists('../cleaned_data'):
        os.makedirs('../cleaned_data')
    
    # Create category directories if they don't exist
    for _, folder_name in CATEGORY_MAPPINGS.items():
        category_dir = f'../cleaned_data/{folder_name}'
        if not os.path.exists(category_dir):
            os.makedirs(category_dir)
    
    return '../cleaned_data'

def generate_category_text(df_category, category_name):
    """Generate formatted text for a specific category"""
    sentences = []
    
    # Add category header
    sentences.append(f"CATEGORY: {category_name.upper()}")
    sentences.append(f"TYPE: OFFERS")
    sentences.append(f"TOTAL ENTRIES: {len(df_category)}")
    sentences.append("-" * 80)
    sentences.append("")  # Empty line after header

    for index, row in df_category.iterrows():
        try:
            offer_name = row.get('OFFER_NAME', 'Unnamed Offer')
            sentence = f"📦 OFFER: {offer_name}\n"
            sentence += "-" * 60 + "\n"
            
            # Add category information explicitly in each offer
            sentence += f"Category: {category_name}\n"

            for raw_col, label in COLUMN_LABELS.items():
                if raw_col == 'OFFER_NAME':
                    continue  # already printed
                value = row.get(raw_col, "I don't have updated information about this.")
                sentence += f"{label}: {value}\n"

            sentences.append(sentence)
        except Exception as e:
            print(f"⚠️ Skipped row {index} due to error: {e}")
    
    return sentences

def clean_offers_data():
    """Clean and process the offers data for all categories"""
    
    # Create directories
    base_dir = create_category_directories()
    
    try:
        # Load the CSV (try UTF-8, fallback to latin1)
        input_file = '../data/offers.csv'
        try:
            df = pd.read_csv(input_file, encoding='utf-8')
        except:
            df = pd.read_csv(input_file, encoding='latin1')

        print(f"📊 Loaded {len(df)} rows from {input_file}")

        # Normalize column names (strip and uppercase)
        df.columns = [col.strip().upper() for col in df.columns]

        # Fill missing values
        df = df.fillna("I don't have updated information about this.")

        # Replace 'null', '-', and empty strings
        for col in df.columns:
            df[col] = df[col].apply(lambda x: "I don't have updated information about this."
                                    if isinstance(x, str) and (x.strip().lower() == 'null' or x.strip() == '-' or x.strip() == '')
                                    else x)

        # Process each category
        for category_header, folder_name in CATEGORY_MAPPINGS.items():
            try:
                # Filter the DataFrame for this category
                df_category = df[df['MAINHEADER'] == category_header]
                
                if len(df_category) == 0:
                    print(f"⚠️ No entries found for category '{category_header}'")
                    continue
                
                # Generate text for this category
                sentences = generate_category_text(df_category, category_header)
                
                # Save to category-specific file
                output_file = f"{base_dir}/{folder_name}/offers.txt"
                with open(output_file, "w", encoding='utf-8') as f:
                    for line in sentences:
                        f.write(line + "\n\n")

                print(f"✅ Processed {len(df_category)} offers for category '{category_header}' and saved to {output_file}")
                
            except Exception as e:
                print(f"❌ Error processing category '{category_header}': {e}")
        
        return True

    except Exception as e:
        print(f"❌ Error processing offers data: {e}")
        return False

if __name__ == "__main__":
    print(f"🔍 Starting cleaning process for offers...")
    clean_offers_data()