"""
utils/helpers.py - Utility functions for JazzBot
"""

import hashlib
import os
import re
from typing import List
from datetime import datetime

from .logger import logger


def compute_file_hash(file_paths: List[str]) -> str:
    """Compute hash of input files"""
    hasher = hashlib.md5()
    for file_path in sorted(file_paths):
        try:
            with open(file_path, "rb") as f:
                hasher.update(f.read())
        except Exception as e:
            logger.error(f"Error hashing {file_path}: {e}")
    return hasher.hexdigest()


def clean_text(text: str) -> str:
    """Clean and normalize text"""
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove special characters but keep important ones
    text = re.sub(r"[^\w\s\-\+\.\,\:\;\(\)\[\]\/\&\%\$\#\@\!]", " ", text)
    return text.strip()


def save_response(question: str, response: str, category: str = None, responses_dir: str = "responses"):
    """Save response to file"""
    try:
        os.makedirs(responses_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        response_file = os.path.join(responses_dir, f"response_{timestamp}.txt")

        with open(response_file, "w", encoding="utf-8") as f:
            f.write(f"Question: {question}\n")
            f.write(f"Response: {response}\n")
            f.write(f"Category: {category or 'Unknown'}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")

    except Exception as e:
        logger.error(f"Error saving response: {e}")


def preprocess_roman_urdu(content: str) -> str:
    """
    Normalize Roman Urdu text by handling common inconsistencies or typos.
    """
    replacements = {
        "krdo": "kar do",
        "acha": "acha",
        "theek": "thik",
        "han": "haan",
        "nai": "nahi",
        # Add more mappings as needed
    }

    for key, value in replacements.items():
        content = content.replace(key, value)
    return content