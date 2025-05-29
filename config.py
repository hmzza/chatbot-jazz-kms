"""
config.py - Configuration settings for JazzBot.

This file contains all configuration constants and settings used across the application.
"""

from dataclasses import dataclass

@dataclass
class Config:
    INDEX_PATH: str = "faiss_index"
    INDEX_HASH_PATH: str = "faiss_index_hash.txt"
    RESPONSES_DIR: str = "responses"
    DATA_DIR: str = "cleaned_data"
    MEMORY_SIZE: int = 5
    CHUNK_SIZE: int = 600
    CHUNK_OVERLAP: int = 80
    SEARCH_K: int = 6
    MAX_RESPONSE_WORDS: int = 150
    STREAM_TIMEOUT: int = 30
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL: str = "llama3"
    LLM_TEMPERATURE: float = 0.1

config = Config()