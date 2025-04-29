
import pandas as pd
import os

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

# Ensure the necessary directory exists
if not os.path.exists('../cleaned_data'):
    os.makedirs('../cleaned_data')

try:
    # Load the CSV (try UTF-8, fallback to latin1)
    try:
        df = pd.read_csv("../data/offers.csv", encoding='utf-8')
    except:
        df = pd.read_csv("../data/offers.csv", encoding='latin1')

    # Normalize column names (strip and uppercase)
    df.columns = [col.strip().upper() for col in df.columns]

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
            offer_name = row.get('OFFER_NAME', 'Unnamed Offer')
            sentence = f"📦 OFFER: {offer_name}\n"
            sentence += "-" * 60 + "\n"

            for raw_col, label in COLUMN_LABELS.items():
                if raw_col == 'OFFER_NAME':
                    continue  # already printed
                value = row.get(raw_col.upper(), "I don't have updated information about this.")
                sentence += f"{label}: {value}\n"

            sentences.append(sentence)
        except Exception as e:
            print(f"⚠️ Skipped row {index} due to error: {e}")

    # Save to file
    output_file = "../cleaned_data/cleaned_offers.txt"
    with open(output_file, "w", encoding='utf-8') as f:
        for line in sentences:
            f.write(line + "\n\n")

    print(f"✅ Done! Cleaned {len(sentences)} offers and saved to {output_file}")

except Exception as e:
    print(f"❌ Error processing offers.csv: {e}")
