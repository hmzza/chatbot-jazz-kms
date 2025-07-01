"""
services/retriever.py - Document retrieval service for JazzBot

This module handles FAISS index creation, document retrieval, and search functionality.
"""

import os
import hashlib
import shutil
from typing import List, Optional, Tuple
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import Config
from models.document import DocumentProcessor
from utils.logger import logger


class RetrieverService:
    """Enhanced document retrieval service using FAISS"""
    
    def __init__(self, config: Config):
        self.config = config
        self.vectorstore = None
        self.embeddings = None
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Initialize HuggingFace embeddings"""
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.config.EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            logger.info("Embeddings initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing embeddings: {e}")
            self.embeddings = None
    
    def compute_file_hash(self, file_paths: List[str]) -> str:
        """Compute hash of input files"""
        hasher = hashlib.md5()
        for file_path in sorted(file_paths):
            try:
                with open(file_path, "rb") as f:
                    hasher.update(f.read())
            except Exception as e:
                logger.error(f"Error hashing {file_path}: {e}")
        return hasher.hexdigest()
    
    def build_retriever(self) -> Tuple[Optional[object], Optional[object]]:
        """Build enhanced FAISS retriever"""
        try:
            docs, file_paths = DocumentProcessor.load_documents_with_categories(self.config.DATA_DIR)

            if not docs:
                logger.warning("No documents found for indexing")
                return None, None

            # Enhanced text splitter
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.CHUNK_SIZE,
                chunk_overlap=self.config.CHUNK_OVERLAP,
                separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
            )

            chunks = splitter.split_documents(docs)
            logger.info(f"Created {len(chunks)} chunks from {len(docs)} documents")

            if not self.embeddings:
                logger.error("Embeddings not available")
                return None, None

            # Check if we can reuse existing index
            current_hash = self.compute_file_hash(file_paths)

            if os.path.exists(self.config.INDEX_PATH) and os.path.exists(self.config.INDEX_HASH_PATH):
                try:
                    with open(self.config.INDEX_HASH_PATH, "r") as f:
                        saved_hash = f.read().strip()

                    if saved_hash == current_hash:
                        logger.info("Reusing existing FAISS index")
                        vectorstore = FAISS.load_local(
                            self.config.INDEX_PATH,
                            self.embeddings,
                            allow_dangerous_deserialization=True,
                        )
                        self.vectorstore = vectorstore
                        return vectorstore, self.embeddings
                except Exception as e:
                    logger.error(f"Error loading existing index: {e}")

            # Build new index
            logger.info("Building new FAISS index...")
            vectorstore = FAISS.from_documents(chunks, self.embeddings)

            # Save index and hash
            vectorstore.save_local(self.config.INDEX_PATH)
            with open(self.config.INDEX_HASH_PATH, "w") as f:
                f.write(current_hash)

            logger.info("FAISS index built and saved successfully")
            self.vectorstore = vectorstore
            return vectorstore, self.embeddings

        except Exception as e:
            logger.error(f"Error building retriever: {e}")
            return None, None
    
    def search_documents(self, query: str, k: int = None) -> List:
        """Search documents using the vectorstore"""
        if not self.vectorstore:
            logger.warning("Vectorstore not available for search")
            return []
        
        try:
            k = k or self.config.SEARCH_K
            docs = self.vectorstore.similarity_search(query, k=k)
            # Filter out discontinued packages
            docs = [
                doc for doc in docs 
                if "discontinued" not in doc.page_content.lower()
            ]
            return docs
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    def rebuild_index(self):
        """Force rebuild of the FAISS index"""
        try:
            # Remove existing index
            if os.path.exists(self.config.INDEX_PATH):
                shutil.rmtree(self.config.INDEX_PATH)
            
            if os.path.exists(self.config.INDEX_HASH_PATH):
                os.remove(self.config.INDEX_HASH_PATH)
            
            # Rebuild
            vectorstore, embeddings = self.build_retriever()
            return vectorstore is not None
        except Exception as e:
            logger.error(f"Error rebuilding index: {e}")
            return False
    
    def get_status(self) -> dict:
        """Get retriever status"""
        return {
            "vectorstore_available": self.vectorstore is not None,
            "embeddings_available": self.embeddings is not None,
            "index_path_exists": os.path.exists(self.config.INDEX_PATH),
            "hash_file_exists": os.path.exists(self.config.INDEX_HASH_PATH),
        }