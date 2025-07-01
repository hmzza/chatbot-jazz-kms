"""
config.py - Configuration settings for JazzBot application
"""

import os
from dataclasses import dataclass


@dataclass
class Config:
    """Application configuration"""
    
    # Paths
    INDEX_PATH: str = "faiss_index"
    INDEX_HASH_PATH: str = "faiss_index_hash.txt"
    RESPONSES_DIR: str = "responses"
    DATA_DIR: str = "cleaned_data"
    
    # Memory and processing
    MEMORY_SIZE: int = 5
    CHUNK_SIZE: int = 600
    CHUNK_OVERLAP: int = 80
    SEARCH_K: int = 6
    MAX_RESPONSE_WORDS: int = 150
    STREAM_TIMEOUT: int = 30
    
    # Models
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL: str = "llama3"
    LLM_TEMPERATURE: float = 0.1
    
    # Flask
    SECRET_KEY: str = os.environ.get("SECRET_KEY", "your_secret_key_here_change_in_production")
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 6061
    
    # Indic NLP
    INDIC_NLP_RESOURCES_PATH: str = "/path-to-indic-nlp-resources"


# Global config instance
config = Config()