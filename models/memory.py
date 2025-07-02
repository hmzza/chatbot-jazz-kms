"""
models/memory.py - Enhanced conversation memory management for JazzBot
"""

import threading
import tiktoken
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from flask import session
from dataclasses import dataclass
from collections import deque

from utils.logger import logger


# Thread-local storage for request context
thread_local = threading.local()


@dataclass
class ConversationEntry:
    """Structured conversation entry"""
    question: str
    answer: str
    category: Optional[str]
    timestamp: datetime
    token_count: int
    importance_score: float = 1.0
    
    def to_dict(self) -> Dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "category": self.category,
            "timestamp": self.timestamp.isoformat(),
            "token_count": self.token_count,
            "importance_score": self.importance_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ConversationEntry':
        return cls(
            question=data["question"],
            answer=data["answer"],
            category=data.get("category"),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            token_count=data.get("token_count", 0),
            importance_score=data.get("importance_score", 1.0)
        )


class TokenManager:
    """Efficient token counting and management"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except:
            # Fallback to cl100k_base for most models
            self.encoding = tiktoken.get_encoding("cl100k_base")
        
        # Cache for token counts to avoid recomputation
        self._token_cache = {}
    
    def count_tokens(self, text: str) -> int:
        """Count tokens with caching"""
        if text in self._token_cache:
            return self._token_cache[text]
        
        try:
            token_count = len(self.encoding.encode(text))
            self._token_cache[text] = token_count
            return token_count
        except:
            # Fallback: rough estimation (4 chars per token)
            return len(text) // 4
    
    def clear_cache(self):
        """Clear token cache to prevent memory buildup"""
        if len(self._token_cache) > 1000:
            self._token_cache.clear()


class ConversationMemory:
    """Enhanced thread-safe conversation memory management with token optimization"""

    def __init__(self, max_tokens: int = 4000, max_entries: int = 20):
        self.max_tokens = max_tokens
        self.max_entries = max_entries
        self.token_manager = TokenManager()
        
        # Importance scoring weights
        self.recency_weight = 0.4
        self.category_weight = 0.3
        self.length_weight = 0.2
        self.follow_up_weight = 0.1
    
    def _is_in_request_context(self) -> bool:
        """Check if we're currently in a Flask request context"""
        try:
            from flask import has_request_context
            return has_request_context()
        except:
            return False

    def _calculate_importance_score(self, entry: ConversationEntry, position: int, total: int) -> float:
        """Calculate importance score for memory prioritization"""
        
        # Recency score (more recent = higher score)
        recency_score = (position / total) if total > 0 else 1.0
        
        # Category importance (specific categories get priority)
        category_score = 1.0
        if entry.category:
            important_categories = ["B2C", "B2B", "data_offers", "packages"]
            category_score = 1.5 if entry.category in important_categories else 1.0
        
        # Length score (more detailed responses get slight priority)
        length_score = min(len(entry.answer) / 200, 2.0)
        
        # Follow-up potential (questions that might lead to follow-ups)
        followup_indicators = ["package", "offer", "plan", "detail", "more"]
        followup_score = 1.5 if any(indicator in entry.question.lower() for indicator in followup_indicators) else 1.0
        
        final_score = (
            recency_score * self.recency_weight +
            category_score * self.category_weight +
            length_score * self.length_weight +
            followup_score * self.follow_up_weight
        )
        
        return final_score

    def get_memory(self) -> List[ConversationEntry]:
        """Get conversation history as structured entries"""
        try:
            # First try thread local (during streaming or out of context)
            if hasattr(thread_local, "chat_memory"):
                return thread_local.chat_memory

            # Only access session if we're in request context
            if self._is_in_request_context():
                try:
                    raw_memory = session.get("chat_memory", [])
                    structured_memory = []
                        
                    for entry_data in raw_memory:
                        try:
                            if isinstance(entry_data, dict) and "timestamp" in entry_data:
                                structured_memory.append(ConversationEntry.from_dict(entry_data))
                            else:
                                # Handle legacy format
                                structured_memory.append(ConversationEntry(
                                    question=entry_data.get("question", ""),
                                    answer=entry_data.get("answer", ""),
                                    category=entry_data.get("category"),
                                    timestamp=datetime.now(),
                                    token_count=self.token_manager.count_tokens(
                                        entry_data.get("question", "") + entry_data.get("answer", "")
                                    )
                                ))
                        except Exception as e:
                            logger.warning(f"Skipping corrupted memory entry: {e}")
                            continue
                        
                    # Cache in thread local for future access
                    thread_local.chat_memory = structured_memory
                    return structured_memory
                except Exception as e:
                    logger.warning(f"Error accessing session memory: {e}")
                
            # Fallback to empty memory
            return []
                
        except Exception as e:
            logger.error(f"Error retrieving memory: {e}")
            return getattr(thread_local, "chat_memory", [])

    def add_exchange(self, question: str, answer: str, category: Optional[str] = None):
        """Add Q&A pair with intelligent memory management"""
        
        # Create structured entry
        token_count = self.token_manager.count_tokens(question + answer)
        entry = ConversationEntry(
            question=question.strip(),
            answer=answer.strip(),
            category=category,
            timestamp=datetime.now(),
            token_count=token_count
        )

        try:
            # Get current memory
            memory = self.get_memory()
            
            # Calculate importance scores
            for i, mem_entry in enumerate(memory):
                mem_entry.importance_score = self._calculate_importance_score(mem_entry, i, len(memory))
            
            # Add new entry
            memory.append(entry)
            
            # Optimize memory based on tokens and importance
            optimized_memory = self._optimize_memory(memory)
            
            # Always update thread local
            thread_local.chat_memory = optimized_memory
            
            # Only update session if we're in request context
            if self._is_in_request_context():
                try:
                    session["chat_memory"] = [entry.to_dict() for entry in optimized_memory]
                    logger.info(f"Memory updated in session - Entries: {len(optimized_memory)}, Tokens: {sum(e.token_count for e in optimized_memory)}")
                except Exception as e:
                    logger.warning(f"Could not update session memory: {e}")
            
            logger.info(f"Added to memory: Q='{question[:50]}...', Tokens={token_count}, Category={category}")

        except Exception as e:
            logger.error(f"Error adding to memory: {e}")
            # Fallback to thread local only
            if not hasattr(thread_local, "chat_memory"):
                thread_local.chat_memory = []
            thread_local.chat_memory.append(entry)

    def _optimize_memory(self, memory: List[ConversationEntry]) -> List[ConversationEntry]:
        """Optimize memory based on token limits and importance scores"""
        
        if not memory:
            return memory
        
        # Always keep the most recent entry
        if len(memory) <= 1:
            return memory
        
        most_recent = memory[-1]
        older_entries = memory[:-1]
        
        # Calculate total tokens
        total_tokens = sum(entry.token_count for entry in memory)
        
        # If within limits, return as is
        if total_tokens <= self.max_tokens and len(memory) <= self.max_entries:
            return memory
        
        # Sort older entries by importance (descending)
        older_entries.sort(key=lambda x: x.importance_score, reverse=True)
        
        # Build optimized memory starting with most recent
        optimized = [most_recent]
        current_tokens = most_recent.token_count
        
        # Add older entries based on importance and token budget
        for entry in older_entries:
            if (current_tokens + entry.token_count <= self.max_tokens and 
                len(optimized) < self.max_entries):
                optimized.insert(-1, entry)  # Insert before most recent
                current_tokens += entry.token_count
            else:
                break
        
        # Sort by timestamp to maintain chronological order
        optimized.sort(key=lambda x: x.timestamp)
        
        return optimized

    def get_context_string(self, max_tokens: Optional[int] = None, include_categories: bool = True) -> str:
        """Generate optimized context string with token awareness"""
        
        memory = self.get_memory()
        if not memory:
            return ""
        
        max_context_tokens = max_tokens or (self.max_tokens // 2)  # Use half of available tokens
        context_parts = []
        current_tokens = 0
        
        # Start from most recent and work backwards
        for i, entry in enumerate(reversed(memory)):
            
            # Format context entry
            if include_categories and entry.category:
                context_line = f"Q{len(memory)-i}: [{entry.category}] {entry.question}\nA{len(memory)-i}: {entry.answer}"
            else:
                context_line = f"Q{len(memory)-i}: {entry.question}\nA{len(memory)-i}: {entry.answer}"
            
            # Check token budget
            line_tokens = self.token_manager.count_tokens(context_line)
            if current_tokens + line_tokens > max_context_tokens:
                break
            
            context_parts.insert(0, context_line)  # Insert at beginning to maintain order
            current_tokens += line_tokens
        
        return "\n\n".join(context_parts)

    def get_relevant_context(self, current_query: str, max_entries: int = 3) -> str:
        """Get context most relevant to current query"""
        
        memory = self.get_memory()
        if not memory:
            return ""
        
        query_lower = current_query.lower()
        query_tokens = set(query_lower.split())
        
        # Score entries based on relevance to current query
        scored_entries = []
        for entry in memory:
            
            # Calculate relevance score
            entry_text = (entry.question + " " + entry.answer).lower()
            entry_tokens = set(entry_text.split())
            
            # Token overlap score
            overlap = len(query_tokens.intersection(entry_tokens))
            overlap_score = overlap / len(query_tokens) if query_tokens else 0
            
            # Category match bonus
            category_bonus = 0.3 if entry.category and entry.category.lower() in query_lower else 0
            
            # Combine with importance score
            final_score = (overlap_score + category_bonus) * entry.importance_score
            
            scored_entries.append((entry, final_score))
        
        # Sort by relevance and take top entries
        scored_entries.sort(key=lambda x: x[1], reverse=True)
        relevant_entries = [entry for entry, score in scored_entries[:max_entries] if score > 0.1]
        
        # Sort selected entries by timestamp for context
        relevant_entries.sort(key=lambda x: x.timestamp)
        
        # Format as context string
        context_parts = []
        for i, entry in enumerate(relevant_entries, 1):
            context_parts.append(f"Relevant Q{i}: {entry.question}")
            context_parts.append(f"Relevant A{i}: {entry.answer}")
        
        return "\n".join(context_parts)

    def get_last_response(self) -> str:
        """Get the last response"""
        memory = self.get_memory()
        return memory[-1].answer if memory else ""

    def get_memory_stats(self) -> Dict:
        """Get memory usage statistics"""
        memory = self.get_memory()
        if not memory:
            return {"total_entries": 0, "total_tokens": 0, "categories": {}}
        
        total_tokens = sum(entry.token_count for entry in memory)
        categories = {}
        
        for entry in memory:
            if entry.category:
                categories[entry.category] = categories.get(entry.category, 0) + 1
        
        return {
            "total_entries": len(memory),
            "total_tokens": total_tokens,
            "token_utilization": total_tokens / self.max_tokens,
            "categories": categories,
            "oldest_entry": memory[0].timestamp.isoformat() if memory else None,
            "newest_entry": memory[-1].timestamp.isoformat() if memory else None
        }

    def cleanup_old_entries(self, days_to_keep: int = 7):
        """Remove entries older than specified days"""
        try:
            memory = self.get_memory()
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            cleaned_memory = [entry for entry in memory if entry.timestamp > cutoff_date]
            
            # Update thread local
            thread_local.chat_memory = cleaned_memory
            
            # Update session if in request context
            if self._is_in_request_context():
                try:
                    session["chat_memory"] = [entry.to_dict() for entry in cleaned_memory]
                except Exception as e:
                    logger.warning(f"Could not update session during cleanup: {e}")
                
            logger.info(f"Cleaned memory: {len(memory) - len(cleaned_memory)} entries removed")
            
        except Exception as e:
            logger.error(f"Error during memory cleanup: {e}")

    def clear_memory(self):
        """Clear all conversation memory"""
        # Clear thread local
        if hasattr(thread_local, "chat_memory"):
            delattr(thread_local, "chat_memory")
        
        # Clear session if in request context
        if self._is_in_request_context():
            try:
                session["chat_memory"] = []
            except Exception as e:
                logger.warning(f"Could not clear session memory: {e}")
        
        self.token_manager.clear_cache()
        logger.info("Memory cleared successfully")