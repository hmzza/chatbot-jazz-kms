"""
models/memory.py - Conversation memory management for JazzBot
"""

import threading
from typing import List, Dict, Optional
from datetime import datetime
from flask import session

from utils.logger import logger


# Thread-local storage for request context
thread_local = threading.local()


class ConversationMemory:
    """Thread-safe conversation memory management"""

    def __init__(self, max_size: int = 5):
        self.max_size = max_size

    def get_memory(self) -> List[Dict]:
        """Get conversation history - thread-safe"""
        try:
            # First try to get from thread local (during streaming)
            if hasattr(thread_local, "chat_memory"):
                return thread_local.chat_memory

            # Otherwise get from session
            if "chat_memory" not in session:
                session["chat_memory"] = []
            return session["chat_memory"]
        except:
            # Fallback to empty memory if session unavailable
            return getattr(thread_local, "chat_memory", [])

    def add_exchange(self, question: str, answer: str, category: Optional[str] = None):
        """Add a Q&A pair to memory - thread-safe"""
        entry = {
            "question": question.strip(),
            "answer": answer.strip(),
            "category": category,
            "timestamp": datetime.now().isoformat(),
        }

        try:
            # Update session memory if available
            memory = session.get("chat_memory", [])
            memory.append(entry)
            if len(memory) > self.max_size:
                memory = memory[-self.max_size :]
            session["chat_memory"] = memory

            # Also update thread local
            thread_local.chat_memory = memory

        except:
            # If session unavailable, use thread local only
            if not hasattr(thread_local, "chat_memory"):
                thread_local.chat_memory = []
            thread_local.chat_memory.append(entry)
            if len(thread_local.chat_memory) > self.max_size:
                thread_local.chat_memory = thread_local.chat_memory[-self.max_size :]

        logger.info(f"Added to memory: Q='{question[:50]}...', Category={category}")

    def get_context_string(self, include_last_n: int = 3) -> str:
        """Generate formatted context string from recent memory"""
        memory = self.get_memory()
        if not memory:
            return ""

        recent_memory = memory[-include_last_n:] if include_last_n > 0 else memory
        context_parts = []

        for i, entry in enumerate(recent_memory, 1):
            context_parts.append(f"Previous Q{i}: {entry['question']}")
            context_parts.append(f"Previous A{i}: {entry['answer']}")

        return "\n".join(context_parts)

    def get_last_response(self) -> str:
        """Get the last response"""
        memory = self.get_memory()
        return memory[-1]["answer"] if memory else ""

    def clear_memory(self):
        """Clear conversation memory"""
        try:
            session["chat_memory"] = []
        except:
            pass
        
        if hasattr(thread_local, "chat_memory"):
            delattr(thread_local, "chat_memory")