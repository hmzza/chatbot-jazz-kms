"""
models/memory.py - Conversation memory management for JazzBot
"""

from typing import List, Optional
from flask import session
from threading import local
from utils.logger import logger

# Thread-local storage
thread_local = local()


class ConversationMemory:
    """Manage conversation history"""

    def __init__(self, max_size: int = 5):
        self.max_size = max_size

    def get_memory(self) -> List[dict]:
        """Get conversation history from session or thread-local"""
        try:
            memory = getattr(
                thread_local, "chat_memory", session.get("chat_memory", [])
            )
            return memory
        except Exception as e:
            logger.error(f"Error accessing memory: {e}")
            return []

    def add_exchange(self, question: str, answer: str, category: str = None):
        """Add a question-answer pair to memory"""
        try:
            memory = self.get_memory()
            memory.append(
                {"question": question, "answer": answer, "category": category}
            )

            if len(memory) > self.max_size:
                memory.pop(0)

            if hasattr(thread_local, "chat_memory"):
                thread_local.chat_memory = memory
            session["chat_memory"] = memory
            logger.info(f"Added exchange to memory: {question} -> {answer}")
        except Exception as e:
            logger.error(f"Error adding exchange to memory: {e}")

    def get_context_string(self, include_last_n: int = 3) -> str:
        """Get formatted context string from recent conversations"""
        try:
            memory = self.get_memory()
            context = []
            for exchange in memory[-include_last_n:]:
                q = exchange.get("question", "")
                a = exchange.get("answer", "")
                c = exchange.get("category", "unknown")
                context.append(f"User: {q}\nAssistant: {a} (Category: {c})")
            return "\n".join(context)
        except Exception as e:
            logger.error(f"Error building context string: {e}")
            return ""

    def get_last_response(self) -> Optional[str]:
        """Get the last response from memory"""
        try:
            memory = self.get_memory()
            return memory[-1]["answer"] if memory else None
        except Exception as e:
            logger.error(f"Error getting last response: {e}")
            return None

    def clear_memory(self):
        """Clear conversation memory"""
        try:
            if hasattr(thread_local, "chat_memory"):
                thread_local.chat_memory = []
            session["chat_memory"] = []
            logger.info("Conversation memory cleared")
        except Exception as e:
            logger.error(f"Error clearing memory: {e}")


# Global instance
memory_manager = ConversationMemory()
