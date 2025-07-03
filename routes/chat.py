"""
routes/chat.py - Chat routes for JazzBot

This module handles all chat-related endpoints and streaming responses.
"""

import time
import threading
from flask import Blueprint, request, Response, session
from langchain_community.llms import Ollama

from config import Config
from models.memory import ConversationMemory
from services.analyzer import QueryAnalyzer, CategoryDetector
from services.prompt import PromptEngine
from services.retriever import RetrieverService
from utils.helpers import save_response, clean_text
from utils.logger import logger
from models.memory import thread_local

# Initialize services
config = Config()
memory_manager = ConversationMemory()
query_analyzer = QueryAnalyzer()
category_detector = CategoryDetector()
prompt_engine = PromptEngine()
retriever_service = RetrieverService(config)

# Initialize LLM
try:
    llm = Ollama(model=config.LLM_MODEL, temperature=config.LLM_TEMPERATURE)
    logger.info("LLM initialized successfully")
except Exception as e:
    logger.error(f"Error initializing LLM: {e}")
    llm = None

# Build retriever on startup
vectorstore, embeddings = retriever_service.build_retriever()

# Create blueprint
chat_bp = Blueprint("chat", __name__)


@chat_bp.route("/chat", methods=["POST"])
def chat():
    """Chat endpoint for English queries"""
    try:
        print("Chat Function Called!!")
        data = request.get_json()
        user_input = data.get("message", "").strip()

        if not user_input:
            return Response("Please enter a message.", mimetype="text/plain")

        if vectorstore is None:
            return Response(
                "Search functionality is currently unavailable. Please try again later.",
                mimetype="text/plain",
            )

        if llm is None:
            return Response(
                "AI service is currently unavailable. Please try again later.",
                mimetype="text/plain",
            )

        logger.info(f"Processing query: {user_input}")

        # Initialize thread local memory
        try:
            if "chat_memory" in session:
                thread_local.chat_memory = session["chat_memory"].copy()
            else:
                thread_local.chat_memory = []
        except:
            thread_local.chat_memory = []

        # Preprocess query
        processed_input = clean_text(user_input)

        # Analyze query
        is_greeting = query_analyzer.is_greeting(processed_input)
        last_response = memory_manager.get_last_response()
        is_followup = query_analyzer.is_followup(processed_input, last_response)
        detected_categories = category_detector.detect_categories(processed_input)
        primary_category = detected_categories[0] if detected_categories else None

        logger.info(
            f"Query analysis - Greeting: {is_greeting}, Follow-up: {is_followup}, "
            f"Categories: {detected_categories}"
        )

        # Handle greeting
        if is_greeting:
            response = prompt_engine.build_greeting_response(user_input)
            return Response(
                _stream_response(response, user_input, None), mimetype="text/plain"
            )

        # Search for relevant documents
        docs = []
        package_name = None

        try:
            if is_followup:
                package_name = query_analyzer.extract_package_name(last_response)
                search_query = (
                    f"{package_name} {primary_category or ''}".strip()
                    if package_name
                    else processed_input
                )
            else:
                search_query = processed_input

            docs = retriever_service.search_documents(search_query, config.SEARCH_K)

            if is_followup and package_name:
                docs = [
                    doc
                    for doc in docs
                    if package_name.lower() in doc.page_content.lower()
                ]

        except Exception as e:
            logger.error(f"Search error: {e}")
            docs = []

        # Build context
        context_parts = []
        for doc in docs:
            if doc.page_content.strip():
                metadata_info = f"[Category: {doc.metadata.get('category', 'unknown')}, Type: {doc.metadata.get('file_type', 'unknown')}]"
                context_parts.append(f"{metadata_info} {doc.page_content}")

        context = "\n\n".join(context_parts)
        memory_context = memory_manager.get_context_string()

        # Generate prompt
        prompt = prompt_engine.build_context_prompt_main(
            context=context,
            memory_context=memory_context,
            user_input=processed_input,
            is_followup=is_followup,
            package_name=package_name,
            detected_categories=detected_categories,
        )

        logger.info(f"Generated prompt length: {len(prompt)} characters")

        return Response(
            _generate_response(prompt, user_input, primary_category),
            mimetype="text/plain",
        )

    except Exception as e:
        logger.error(f"Chat error: {e}")
        error_msg = (
            "I apologize, but I encountered an unexpected error. Please try again."
        )
        return Response(_stream_text(error_msg), mimetype="text/plain")


def _generate_response(prompt: str, user_input: str, primary_category: str):
    """Generate streaming response"""
    print("Generate Function Called!!")
    full_response = []
    start_time = time.time()

    try:
        try:
            if hasattr(llm, "stream"):
                for chunk in llm.stream(prompt):
                    if chunk:
                        full_response.append(chunk)
                        yield chunk
                        time.sleep(0.02)
            else:
                response = llm.invoke(prompt)
                full_response.append(response)
                words = response.split()
                for i, word in enumerate(words):
                    if i == 0:
                        yield word
                    else:
                        yield f" {word}"
                    time.sleep(0.08)

        except Exception as stream_error:
            logger.warning(f"Streaming failed, using invoke: {stream_error}")
            response = llm.invoke(prompt)
            full_response.append(response)

            for char in response:
                yield char
                if char in [" ", ".", "!", "?", "\n"]:
                    time.sleep(0.05)
                else:
                    time.sleep(0.01)

    except Exception as e:
        logger.error(f"Error in generation: {e}")
        error_msg = "I apologize, but I encountered an error processing your request. Please try again."
        full_response = [error_msg]
        for char in error_msg:
            yield char
            time.sleep(0.03)
        return

    complete_response = "".join(full_response).strip()

    # Update memory in background thread
    def update_memory():
        try:
            memory_manager.add_exchange(user_input, complete_response, primary_category)
            save_response(user_input, complete_response, primary_category)
        except Exception as e:
            logger.error(f"Error updating memory: {e}")

    threading.Thread(target=update_memory, daemon=True).start()


def _stream_response(response: str, user_input: str, category: str):
    """Stream a pre-generated response"""
    words = response.split()
    for i, word in enumerate(words):
        if i == 0:
            yield word
        else:
            yield f" {word}"
        time.sleep(0.05)

    # Update memory
    memory_manager.add_exchange(user_input, response, category)


def _stream_text(text: str):
    """Stream text character by character"""
    for char in text:
        yield char
        time.sleep(0.03)


@chat_bp.route("/clear_memory", methods=["POST"])
def clear_memory():
    """Clear conversation memory"""
    try:
        memory_manager.clear_memory()
        return Response("Memory cleared successfully", status=200)
    except Exception as e:
        logger.error(f"Error clearing memory: {e}")
        return Response("Failed to clear memory", status=500)


# Make services available for other modules
def get_retriever_service():
    return retriever_service


def get_llm():
    return llm
