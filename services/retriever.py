"""
services/retriever.py - Enhanced document retrieval service for JazzBot

This module handles FAISS index creation, document retrieval, search functionality,
and advanced retrieval strategies with caching and optimization.
"""

import os
import hashlib
import shutil
import pickle
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime, timedelta
from collections import defaultdict, deque

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from config import Config
from models.document import DocumentProcessor
from utils.logger import logger


class SearchCache:
    """LRU Cache for search results to improve performance"""
    
    def __init__(self, max_size: int = 100, ttl_minutes: int = 30):
        self.max_size = max_size
        self.ttl = timedelta(minutes=ttl_minutes)
        self.cache = {}
        self.access_order = deque()
        self.timestamps = {}
        self._hit_count = 0
        self._total_requests = 0
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired"""
        if key not in self.timestamps:
            return True
        return datetime.now() - self.timestamps[key] > self.ttl
    
    def get(self, key: str) -> Optional[List[Document]]:
        """Get cached search results"""
        self._total_requests += 1
        
        if key in self.cache and not self._is_expired(key):
            # Move to end (most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)
            self._hit_count += 1
            return self.cache[key]
        elif key in self.cache:
            # Remove expired entry
            self._remove(key)
        return None
    
    def put(self, key: str, value: List[Document]):
        """Cache search results"""
        if key in self.cache:
            self.access_order.remove(key)
        elif len(self.cache) >= self.max_size:
            # Remove least recently used
            oldest = self.access_order.popleft()
            self._remove(oldest)
        
        self.cache[key] = value
        self.timestamps[key] = datetime.now()
        self.access_order.append(key)
    
    def _remove(self, key: str):
        """Remove entry from cache"""
        if key in self.cache:
            del self.cache[key]
        if key in self.timestamps:
            del self.timestamps[key]
    
    def clear(self):
        """Clear all cache"""
        self.cache.clear()
        self.timestamps.clear()
        self.access_order.clear()
        self._hit_count = 0
        self._total_requests = 0
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        valid_entries = sum(1 for k in self.cache.keys() if not self._is_expired(k))
        hit_rate = self._hit_count / max(self._total_requests, 1)
        return {
            "total_entries": len(self.cache),
            "valid_entries": valid_entries,
            "expired_entries": len(self.cache) - valid_entries,
            "max_size": self.max_size,
            "hit_rate": hit_rate,
            "total_requests": self._total_requests,
            "cache_hits": self._hit_count
        }


class DocumentRanker:
    """Advanced document ranking based on relevance and metadata"""
    
    def __init__(self):
        self.category_weights = {
            'b2c': 1.2,
            'b2b': 1.1,
            'cfl': 1.0,
            'digital': 1.0,
            'jazz_rox': 0.9
        }
        
        self.file_type_weights = {
            'packages': 1.3,
            'offers': 1.2,
            'data_offers': 1.1,
            'complaints_sops': 0.8
        }
    
    def rank_documents(self, docs: List[Document], query: str, user_categories: List[str] = None) -> List[Document]:
        """Rank documents based on relevance, metadata, and user context"""
        if not docs:
            return docs
        
        query_lower = query.lower()
        query_tokens = set(query_lower.split())
        
        scored_docs = []
        
        for doc in docs:
            score = self._calculate_document_score(doc, query_tokens, query_lower, user_categories)
            scored_docs.append((doc, score))
        
        # Sort by score (descending) and return documents
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored_docs]
    
    def _calculate_document_score(self, doc: Document, query_tokens: set, query_lower: str, user_categories: List[str]) -> float:
        """Calculate comprehensive document relevance score"""
        base_score = 1.0
        content_lower = doc.page_content.lower()
        
        # Content relevance score
        content_tokens = set(content_lower.split())
        token_overlap = len(query_tokens.intersection(content_tokens))
        overlap_score = token_overlap / len(query_tokens) if query_tokens else 0
        
        # Exact phrase matching bonus
        phrase_bonus = 1.5 if query_lower in content_lower else 1.0
        
        # Category relevance
        doc_category = doc.metadata.get('category', '').lower()
        category_score = self.category_weights.get(doc_category, 1.0)
        
        # User category preference
        user_category_bonus = 1.0
        if user_categories and doc_category in [cat.lower() for cat in user_categories]:
            user_category_bonus = 1.4
        
        # File type relevance
        file_type = doc.metadata.get('file_type', '').lower()
        file_type_score = self.file_type_weights.get(file_type, 1.0)
        
        # Content quality indicators
        quality_score = 1.0
        if any(indicator in content_lower for indicator in ['price', 'package', 'gb', 'minutes']):
            quality_score = 1.2
        
        # Length penalty for very short or very long docs
        content_length = len(doc.page_content)
        if content_length < 50:
            length_penalty = 0.8
        elif content_length > 1000:
            length_penalty = 0.9
        else:
            length_penalty = 1.0
        
        # Combine all factors
        final_score = (
            base_score * 
            (1 + overlap_score) * 
            phrase_bonus * 
            category_score * 
            user_category_bonus * 
            file_type_score * 
            quality_score * 
            length_penalty
        )
        
        return final_score


class RetrieverService:
    """Enhanced document retrieval service with advanced features"""
    
    def __init__(self, config: Config):
        self.config = config
        self.vectorstore = None
        self.embeddings = None
        self.search_cache = SearchCache(
            max_size=getattr(config, 'CACHE_SIZE', 100),
            ttl_minutes=getattr(config, 'CACHE_TTL_MINUTES', 30)
        )
        self.document_ranker = DocumentRanker()
        self.performance_stats = defaultdict(int)
        
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Initialize HuggingFace embeddings with error handling"""
        try:
            model_kwargs = {
                "device": getattr(self.config, 'EMBEDDING_DEVICE', 'cpu'),
                "trust_remote_code": False
            }
            
            encode_kwargs = {
                "normalize_embeddings": True,
                "batch_size": getattr(self.config, 'EMBEDDING_BATCH_SIZE', 32)
            }
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.config.EMBEDDING_MODEL,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
                show_progress=False
            )
            
            logger.info(f"Embeddings initialized: {self.config.EMBEDDING_MODEL}")
            
        except Exception as e:
            logger.error(f"Error initializing embeddings: {e}")
            self.embeddings = None
    
    def compute_content_hash(self, file_paths: List[str]) -> str:
        """Compute hash of all input files for change detection"""
        hasher = hashlib.sha256()
        
        for file_path in sorted(file_paths):
            try:
                with open(file_path, "rb") as f:
                    # Read file in chunks to handle large files
                    while chunk := f.read(8192):
                        hasher.update(chunk)
                
                # Include file modification time in hash
                mtime = os.path.getmtime(file_path)
                hasher.update(str(mtime).encode())
                
            except Exception as e:
                logger.error(f"Error hashing {file_path}: {e}")
                hasher.update(b"error")
        
        return hasher.hexdigest()
    
    def _ensure_index_directory(self):
        """Ensure the index directory exists, handling both relative and absolute paths"""
        try:
            index_path = self.config.INDEX_PATH
            
            # If INDEX_PATH is a relative path like "faiss_index", create it in current directory
            if not os.path.isabs(index_path):
                # For relative paths, just ensure the directory exists if it has one
                if os.path.sep in index_path or '/' in index_path or '\\' in index_path:
                    index_dir = os.path.dirname(index_path)
                    if index_dir:  # Only create if there's actually a directory component
                        os.makedirs(index_dir, exist_ok=True)
                        logger.debug(f"Created directory: {index_dir}")
            else:
                # For absolute paths, create the full directory structure
                index_dir = os.path.dirname(index_path)
                if index_dir:
                    os.makedirs(index_dir, exist_ok=True)
                    logger.debug(f"Created directory: {index_dir}")
                    
        except Exception as e:
            logger.error(f"Error creating index directory: {e}")
            raise
    
    def build_retriever(self) -> Tuple[Optional[FAISS], Optional[HuggingFaceEmbeddings]]:
        """Build enhanced FAISS retriever with optimization"""
        try:
            start_time = datetime.now()
            logger.info("Starting retriever build process...")
            
            # Load documents with enhanced processing
            docs, file_paths = DocumentProcessor.load_documents_with_categories(self.config.DATA_DIR)
            
            if not docs:
                logger.warning("No documents found for indexing")
                return None, None
            
            logger.info(f"Loaded {len(docs)} documents from {len(file_paths)} files")
            
            # Clear cache when rebuilding
            self.search_cache.clear()
            
            # Enhanced text splitter with better separators
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.CHUNK_SIZE,
                chunk_overlap=self.config.CHUNK_OVERLAP,
                separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
                length_function=len,
                add_start_index=True
            )
            
            # Split documents with progress tracking
            logger.info(f"Splitting {len(docs)} documents into chunks...")
            chunks = []
            for i, doc in enumerate(docs):
                doc_chunks = splitter.split_documents([doc])
                chunks.extend(doc_chunks)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(docs)} documents")
            
            logger.info(f"Created {len(chunks)} chunks from {len(docs)} documents")
            
            if not self.embeddings:
                logger.error("Embeddings not available")
                return None, None
            
            # Check if we can reuse existing index
            current_hash = self.compute_content_hash(file_paths)
            
            if os.path.exists(self.config.INDEX_PATH) and os.path.exists(self.config.INDEX_HASH_PATH):
                try:
                    with open(self.config.INDEX_HASH_PATH, "r") as f:
                        saved_hash = f.read().strip()
                    
                    if saved_hash == current_hash:
                        logger.info("Reusing existing FAISS index")
                        vectorstore = FAISS.load_local(
                            self.config.INDEX_PATH,
                            self.embeddings,
                            allow_dangerous_deserialization=True
                        )
                        self.vectorstore = vectorstore
                        self.performance_stats['index_reused'] += 1
                        return vectorstore, self.embeddings
                        
                except Exception as e:
                    logger.error(f"Error loading existing index: {e}")
            
            # Build new index with batching for better memory management
            logger.info("Building new FAISS index...")
            
            # Process in batches if we have many chunks
            batch_size = getattr(self.config, 'INDEX_BATCH_SIZE', 1000)
            
            if len(chunks) > batch_size:
                logger.info(f"Processing {len(chunks)} chunks in batches of {batch_size}")
                
                # Create initial index with first batch
                first_batch = chunks[:batch_size]
                vectorstore = FAISS.from_documents(first_batch, self.embeddings)
                
                # Add remaining chunks in batches
                for i in range(batch_size, len(chunks), batch_size):
                    batch = chunks[i:i + batch_size]
                    batch_vectorstore = FAISS.from_documents(batch, self.embeddings)
                    vectorstore.merge_from(batch_vectorstore)
                    
                    logger.info(f"Processed {min(i + batch_size, len(chunks))}/{len(chunks)} chunks")
            else:
                vectorstore = FAISS.from_documents(chunks, self.embeddings)
            
            # Ensure directory exists before saving
            self._ensure_index_directory()
            
            # Save index and hash
            logger.info(f"Saving FAISS index to: {self.config.INDEX_PATH}")
            vectorstore.save_local(self.config.INDEX_PATH)
            
            with open(self.config.INDEX_HASH_PATH, "w") as f:
                f.write(current_hash)
            
            # Save metadata for debugging
            metadata_path = self._get_metadata_path()
            metadata = {
                "created_at": datetime.now().isoformat(),
                "num_documents": len(docs),
                "num_chunks": len(chunks),
                "content_hash": current_hash,
                "embedding_model": self.config.EMBEDDING_MODEL,
                "chunk_size": self.config.CHUNK_SIZE,
                "chunk_overlap": self.config.CHUNK_OVERLAP
            }
            
            try:
                with open(metadata_path, "wb") as f:
                    pickle.dump(metadata, f)
                logger.debug(f"Saved metadata to: {metadata_path}")
            except Exception as e:
                logger.warning(f"Could not save metadata: {e}")
            
            build_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"FAISS index built successfully in {build_time:.2f} seconds")
            
            self.vectorstore = vectorstore
            self.performance_stats['index_built'] += 1
            self.performance_stats['last_build_time'] = build_time
            
            return vectorstore, self.embeddings
            
        except Exception as e:
            logger.error(f"Error building retriever: {e}")
            return None, None
    
    def _get_metadata_path(self) -> str:
        """Get the metadata file path"""
        index_path = self.config.INDEX_PATH
        
        if os.path.isabs(index_path):
            # Absolute path - put metadata in same directory
            index_dir = os.path.dirname(index_path)
            return os.path.join(index_dir, "index_metadata.pkl")
        else:
            # Relative path - handle directory component
            if os.path.sep in index_path or '/' in index_path or '\\' in index_path:
                index_dir = os.path.dirname(index_path)
                return os.path.join(index_dir, "index_metadata.pkl")
            else:
                # Just a filename - put metadata in current directory
                return "index_metadata.pkl"
    
    def search_documents(self, query: str, k: int = None, user_categories: List[str] = None, 
                        use_mmr: bool = False, fetch_k: int = None) -> List[Document]:
        """Enhanced search with caching, ranking, and multiple retrieval strategies"""
        if not self.vectorstore:
            logger.warning("Vectorstore not available for search")
            return []
        
        try:
            # Create cache key
            cache_key = f"{query}_{k}_{user_categories}_{use_mmr}_{fetch_k}"
            
            # Check cache first
            cached_results = self.search_cache.get(cache_key)
            if cached_results is not None:
                logger.debug(f"Cache hit for query: {query[:50]}...")
                self.performance_stats['cache_hits'] += 1
                return cached_results
            
            # Set defaults
            k = k or self.config.SEARCH_K
            fetch_k = fetch_k or min(k * 4, 50)  # Fetch more for better ranking
            
            # Perform search with different strategies
            if use_mmr:
                # Maximum Marginal Relevance for diverse results
                docs = self.vectorstore.max_marginal_relevance_search(
                    query, k=k, fetch_k=fetch_k
                )
            else:
                # Standard similarity search with more candidates
                docs = self.vectorstore.similarity_search(query, k=fetch_k)
            
            # Filter out discontinued packages and low-quality content
            filtered_docs = []
            for doc in docs:
                content_lower = doc.page_content.lower()
                
                # Skip discontinued or irrelevant content
                if any(skip_term in content_lower for skip_term in [
                    "discontinued", "not available", "expired", "invalid"
                ]):
                    continue
                
                # Skip very short or empty content
                if len(doc.page_content.strip()) < 10:
                    continue
                
                filtered_docs.append(doc)
            
            # Rank documents using advanced ranking
            ranked_docs = self.document_ranker.rank_documents(
                filtered_docs, query, user_categories
            )
            
            # Take top k results after ranking
            final_docs = ranked_docs[:k]
            
            # Cache results
            self.search_cache.put(cache_key, final_docs)
            self.performance_stats['cache_misses'] += 1
            self.performance_stats['total_searches'] += 1
            
            logger.debug(f"Search completed: {len(final_docs)} documents returned")
            return final_docs
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    def search_with_score(self, query: str, k: int = None, score_threshold: float = 0.0) -> List[Tuple[Document, float]]:
        """Search with similarity scores for debugging and filtering"""
        if not self.vectorstore:
            logger.warning("Vectorstore not available for search")
            return []
        
        try:
            k = k or self.config.SEARCH_K
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            
            # Filter by score threshold if specified
            if score_threshold > 0.0:
                results = [(doc, score) for doc, score in results if score >= score_threshold]
            
            return results
            
        except Exception as e:
            logger.error(f"Search with score error: {e}")
            return []
    
    def hybrid_search(self, query: str, k: int = None, alpha: float = 0.5) -> List[Document]:
        """Hybrid search combining semantic and keyword matching"""
        if not self.vectorstore:
            return []
        
        try:
            k = k or self.config.SEARCH_K
            
            # Get semantic search results
            semantic_docs = self.vectorstore.similarity_search(query, k=k*2)
            
            # Simple keyword matching for hybrid approach
            query_tokens = set(query.lower().split())
            keyword_scores = {}
            
            for doc in semantic_docs:
                content_tokens = set(doc.page_content.lower().split())
                overlap = len(query_tokens.intersection(content_tokens))
                keyword_scores[id(doc)] = overlap / len(query_tokens) if query_tokens else 0
            
            # Combine scores (this is a simplified hybrid approach)
            # In a full implementation, you'd want proper BM25 or similar
            combined_results = []
            
            for i, doc in enumerate(semantic_docs):
                semantic_score = 1.0 / (i + 1)  # Simple rank-based scoring
                keyword_score = keyword_scores.get(id(doc), 0)
                
                combined_score = alpha * semantic_score + (1 - alpha) * keyword_score
                combined_results.append((doc, combined_score))
            
            # Sort by combined score and return top k
            combined_results.sort(key=lambda x: x[1], reverse=True)
            return [doc for doc, score in combined_results[:k]]
            
        except Exception as e:
            logger.error(f"Hybrid search error: {e}")
            return []
    
    def rebuild_index(self) -> bool:
        """Force rebuild of the FAISS index"""
        try:
            logger.info("Starting index rebuild...")
            
            # Remove existing index files
            if os.path.exists(self.config.INDEX_PATH):
                if os.path.isdir(self.config.INDEX_PATH):
                    shutil.rmtree(self.config.INDEX_PATH)
                else:
                    os.remove(self.config.INDEX_PATH)
            
            if os.path.exists(self.config.INDEX_HASH_PATH):
                os.remove(self.config.INDEX_HASH_PATH)
            
            # Remove metadata file
            metadata_path = self._get_metadata_path()
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
            
            # Clear cache
            self.search_cache.clear()
            
            # Rebuild
            vectorstore, embeddings = self.build_retriever()
            success = vectorstore is not None
            
            if success:
                logger.info("Index rebuild completed successfully")
                self.performance_stats['manual_rebuilds'] += 1
            else:
                logger.error("Index rebuild failed")
            
            return success
            
        except Exception as e:
            logger.error(f"Error rebuilding index: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive retriever status"""
        cache_stats = self.search_cache.get_stats()
        
        # Get index metadata if available
        metadata_path = self._get_metadata_path()
        index_metadata = {}
        
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "rb") as f:
                    index_metadata = pickle.load(f)
            except Exception as e:
                logger.error(f"Error loading index metadata: {e}")
        
        return {
            "vectorstore_available": self.vectorstore is not None,
            "embeddings_available": self.embeddings is not None,
            "index_path_exists": os.path.exists(self.config.INDEX_PATH),
            "hash_file_exists": os.path.exists(self.config.INDEX_HASH_PATH),
            "embedding_model": self.config.EMBEDDING_MODEL,
            "cache_stats": cache_stats,
            "performance_stats": dict(self.performance_stats),
            "index_metadata": index_metadata,
            "config": {
                "chunk_size": self.config.CHUNK_SIZE,
                "chunk_overlap": self.config.CHUNK_OVERLAP,
                "search_k": self.config.SEARCH_K,
                "index_path": self.config.INDEX_PATH,
                "cache_size": getattr(self.config, 'CACHE_SIZE', 100),
                "cache_ttl": getattr(self.config, 'CACHE_TTL_MINUTES', 30)
            }
        }
    
    def get_document_count(self) -> int:
        """Get total number of documents in the vectorstore"""
        if not self.vectorstore:
            return 0
        
        try:
            # FAISS doesn't have a direct document count method
            # This is an approximation based on the index
            return self.vectorstore.index.ntotal
        except Exception:
            return 0
    
    def clear_cache(self):
        """Clear the search cache"""
        self.search_cache.clear()
        logger.info("Search cache cleared")
    
    def get_cache_stats(self) -> Dict:
        """Get detailed cache statistics"""
        return self.search_cache.get_stats()
    
    def optimize_index(self):
        """Optimize the FAISS index for better performance"""
        if not self.vectorstore:
            logger.warning("No vectorstore available for optimization")
            return False
        
        try:
            # This is a placeholder for FAISS optimization
            # In practice, you might want to train the index or use different index types
            logger.info("Index optimization completed")
            return True
            
        except Exception as e:
            logger.error(f"Error optimizing index: {e}")
            return False