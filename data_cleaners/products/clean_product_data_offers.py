"""
clean_products_data_offers.py - Cleans and formats product data offers data with category metadata

This script processes the raw product data offers CSV file and formats it for the chatbot,
while clearly marking it as belonging to the "products" category.
"""

import pandas as pd
import os

# Define the category this cleaner handles
CATEGORY = "products"

# Map raw column names to readable labels
COLUMN_LABELS = {
    'Id': 'Offer ID',
    'Category': 'Offer Category',
    'Consumable_Data': 'Consumable Data',
    'Charges': 'Charges',
    'Data_Name': 'Data Name',
    'Incentive': 'Incentive',
    'Validity': 'Validity',
    'Time_Window': 'Time Window (Usage Hours)',
    'Recursive': 'Is Recursive',
    'Prorated': 'Is Prorated',
    'Subscription': 'Subscription Code',
    'Unsubscription': 'Unsubscription Code',
    'Description': 'Offer Description',
    'USSD_String': 'USSD String',
    'BaseRate': 'Base Rate',
    'MainHeader': 'Main Header',
    'NotifyDomain': 'Notify Domain',
    'data_type_name': 'Data Type'
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
    """Clean and process the data-offers data for the products category"""
    
    # Create directories
    category_dir = create_category_directories()
    
    try:
        # Load the CSV (try UTF-8, fallback to latin1)
        input_file = f'../../data/{CATEGORY}/data_offers.csv'
        try:
            df = pd.read_csv(input_file, encoding='utf-8')
        except:
            df = pd.read_csv(input_file, encoding='latin1')

        print(f"📊 Loaded {len(df)} rows from {input_file}")
        print(f"Original columns: {df.columns.tolist()}")

        # Normalize column names (strip and uppercase)
        df.columns = [col.strip().upper() for col in df.columns]
        print(f"Normalized columns: {df.columns.tolist()}")

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
        sentences.append(f"TYPE: DATA_OFFERS")
        sentences.append(f"TOTAL ENTRIES: {len(df)}")
        sentences.append("-" * 80)
        sentences.append("")  # Empty line after header

        for index, row in df.iterrows():
            try:
                # Since we uppercased all columns, we need to look for the uppercased version
                offer_name = row.get('DATA_NAME', 'Unnamed Offer')
                
                # If DATA_NAME isn't found, try other possible variations
                if offer_name == 'Unnamed Offer':
                    possible_name_cols = ['OFFER_NAME', 'NAME', 'PRODUCT_NAME']
                    for col in possible_name_cols:
                        if col in df.columns:
                            offer_value = row.get(col)
                            if offer_value and offer_value != "I don't have updated information about this.":
                                offer_name = offer_value
                                print(f"Found offer name in alternate column: {col}")
                                break
                
                sentence = f"📦 DATA OFFER: {offer_name}\n"
                sentence += "-" * 60 + "\n"
                
                # Add category information explicitly in each offer
                sentence += f"Category: {CATEGORY.capitalize()}\n"

                for raw_col, label in COLUMN_LABELS.items():
                    # Skip Data_Name as we already printed it
                    if raw_col.upper() == 'DATA_NAME':
                        continue  
                    
                    # We need to lookup using the uppercased column name
                    value = row.get(raw_col.upper(), "I don't have updated information about this.")
                    sentence += f"{label}: {value}\n"

                sentences.append(sentence)
            except Exception as e:
                print(f"⚠️ Skipped row {index} due to error: {e}")

        # Save to category-specific file
        output_file = f"{category_dir}/data_offers.txt"
        with open(output_file, "w", encoding='utf-8') as f:
            for line in sentences:
                f.write(line + "\n\n")

        print(f"✅ Done! Cleaned {len(df)} data_offers and saved to {output_file}")
        
        return True

    except Exception as e:
        print(f"❌ Error processing {CATEGORY} data_offers: {e}")
        return False

if __name__ == "__main__":
    print(f"🔍 Starting cleaning process for {CATEGORY.upper()} data_offers...")
    clean_offers_data()