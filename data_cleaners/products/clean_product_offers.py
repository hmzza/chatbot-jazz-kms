"""
clean_products_offers.py - Cleans and formats product offers data with category metadata

This script processes the raw product offers CSV file and formats it for the chatbot,
while clearly marking it as belonging to the "products" category.
"""

import pandas as pd
import os

# Define the category this cleaner handles
CATEGORY = "products"

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
    'OfferType': 'Offer Type',
    'MainHeader': 'Main Header'
}

def create_category_directories():
    """Ensure the necessary directory structure exists"""
    
    # Create base directory if it doesn't exist
    if not os.path.exists('../../cleaned_data'):
        os.makedirs('../../cleaned_data')
    
    # Create category directory if it doesn't exist
    category_dir = f'../../cleaned_data/{CATEGORY}'
    if not os.path.exists(category_dir):
        os.makedirs(category_dir)
    
    return category_dir

def clean_offers_data():
    """Clean and process the offers data for the products category"""
    
    # Create directories
    category_dir = create_category_directories()
    
    try:
        # Load the CSV (try UTF-8, fallback to latin1)
        input_file = f'../../data/{CATEGORY}/offers.csv'
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

        # Generate text sentences with category metadata
        sentences = []
        
        # Add category header
        sentences.append(f"CATEGORY: {CATEGORY.upper()}")
        sentences.append(f"TYPE: OFFERS")
        sentences.append(f"TOTAL ENTRIES: {len(df)}")
        sentences.append("-" * 80)
        sentences.append("")  # Empty line after header

        for index, row in df.iterrows():
            try:
                offer_name = row.get('OFFER_NAME', 'Unnamed Offer')
                sentence = f"📦 OFFER: {offer_name}\n"
                sentence += "-" * 60 + "\n"
                
                # Add category information explicitly in each offer
                sentence += f"Category: {CATEGORY.capitalize()}\n"

                for raw_col, label in COLUMN_LABELS.items():
                    if raw_col == 'OFFER_NAME':
                        continue  # already printed
                    value = row.get(raw_col.upper(), "I don't have updated information about this.")
                    sentence += f"{label}: {value}\n"

                sentences.append(sentence)
            except Exception as e:
                print(f"⚠️ Skipped row {index} due to error: {e}")

        # Save to category-specific file
        output_file = f"{category_dir}/offers.txt"
        with open(output_file, "w", encoding='utf-8') as f:
            for line in sentences:
                f.write(line + "\n\n")

        print(f"✅ Done! Cleaned {len(df)} offers and saved to {output_file}")
        
        return True

    except Exception as e:
        print(f"❌ Error processing {CATEGORY} offers: {e}")
        return False

if __name__ == "__main__":
    print(f"🔍 Starting cleaning process for {CATEGORY.upper()} offers...")
    clean_offers_data()