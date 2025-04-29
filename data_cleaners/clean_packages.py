import pandas as pd
import os

# Map raw column names to readable labels
COLUMN_LABELS = {
    'Type': 'Package Type',
    'MainType': 'Main Type',
    'Network': 'Network',
    'Pack_Name': 'Package Name',
    'ONNET_CALLS_BASERATE': 'On-Network Calls Base Rate',
    'ONNET_CALLS_TAX': 'On-Network Calls Tax',
    'OFFNET_CALLS_BASERATE': 'Off-Network Calls Base Rate',
    'OFFNET_CALLS_TAX': 'Off-Network Calls Tax',
    'ONNET_SMS_BASERATE': 'On-Network SMS Base Rate',
    'ONNET_SMS_TAX': 'On-Network SMS Tax',
    'OFFNET_SMS_BASERATE': 'Off-Network SMS Base Rate',
    'OFFNET_SMS_TAX': 'Off-Network SMS Tax',
    'Intl_SMS_BaseRate': 'International SMS Base Rate',
    'Intl_SMS_Tax': 'International SMS Tax',
    'CreatedBy': 'Created By',
    'URL': 'URL',
    'FNF_Numbers': 'Friends & Family Numbers',
    'USSD_String': 'USSD String',
    'ViewType': 'View Type',
    'Complaints': 'Complaints',
    'MainHeader': 'Main Header',
    'NotifyDomain': 'Notify Domain'
}

# Ensure the necessary directory exists
if not os.path.exists('../cleaned_data'):
    os.makedirs('../cleaned_data')

try:
    # Load the CSV (try UTF-8, fallback to latin1)
    try:
        df = pd.read_csv("../data/packages.csv", encoding='utf-8')
    except:
        df = pd.read_csv("../data/packages.csv", encoding='latin1')
    
    # Normalize column names (strip)
    df.columns = [col.strip() for col in df.columns]
    
    # Fill missing values
    df = df.fillna("I don't have updated information about this.")
    
    # Replace 'null', '-', and empty strings
    for col in df.columns:
        df[col] = df[col].apply(lambda x: "I don't have updated information about this."
                            if isinstance(x, str) and (x.strip().lower() == 'null' or x.strip() == '-' or x.strip() == '')
                            else x)
    
    # Generate text sentences
    sentences = []
    
    for index, row in df.iterrows():
        try:
            package_name = row.get('Pack_Name', 'Unnamed Package')
            sentence = f"📱 PACKAGE: {package_name}\n"
            sentence += "-" * 60 + "\n"
            
            for raw_col, label in COLUMN_LABELS.items():
                if raw_col == 'Pack_Name':
                    continue  # already printed
                value = row.get(raw_col, "I don't have updated information about this.")
                sentence += f"{label}: {value}\n"
            
            sentences.append(sentence)
        except Exception as e:
            print(f"⚠️ Skipped row {index} due to error: {e}")
    
    # Save to file
    output_file = "../cleaned_data/cleaned_packages.txt"
    with open(output_file, "w", encoding='utf-8') as f:
        for line in sentences:
            f.write(line + "\n\n")
    
    print(f"✅ Done! Cleaned {len(sentences)} packages and saved to {output_file}")

except Exception as e:
    print(f"❌ Error processing raw_packages.csv: {e}")