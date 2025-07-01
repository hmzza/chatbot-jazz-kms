"""
models/document.py - Document processing for JazzBot
"""

import os
from typing import List, Tuple
from langchain_community.document_loaders import TextLoader

from utils.helpers import clean_text, preprocess_roman_urdu
from utils.logger import logger


class DocumentProcessor:
    """Enhanced document processing and retrieval"""

    @staticmethod
    def load_documents_with_categories(data_dir: str) -> Tuple[List, List[str]]:
        """Load documents with enhanced metadata"""
        docs = []
        file_paths = []

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            logger.warning(f"Created {data_dir} directory - no documents found")
            return docs, file_paths

        for category in os.listdir(data_dir):
            category_path = os.path.join(data_dir, category)
            if not os.path.isdir(category_path):
                continue

            for file in os.listdir(category_path):
                if not file.endswith(".txt"):
                    continue

                file_path = os.path.join(category_path, file)
                file_paths.append(file_path)

                # Determine file type with better logic
                file_type = "unknown"
                file_lower = file.lower()
                if "offer" in file_lower:
                    file_type = "offers"
                elif "package" in file_lower:
                    file_type = "packages"
                elif "data" in file_lower:
                    file_type = "data_offers"
                elif "bundle" in file_lower:
                    file_type = "bundles"

                try:
                    loader = TextLoader(file_path, encoding="utf-8")
                    documents = loader.load()

                    for doc in documents:
                        # Preprocess for Roman Urdu
                        doc.page_content = preprocess_roman_urdu(
                            clean_text(doc.page_content)
                        )
                        doc.metadata.update(
                            {
                                "category": category,
                                "file_type": file_type,
                                "filename": file,
                                "source_path": file_path,
                                "is_roman_urdu": "Roman Urdu" in category.lower(),
                            }
                        )

                    docs.extend(documents)
                    logger.info(f"Loaded {len(documents)} documents from {file_path}")

                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")

        logger.info(f"Total documents loaded: {len(docs)}")
        return docs, file_paths
