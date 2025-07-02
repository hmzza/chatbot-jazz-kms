"""
routes/chat.py - Enhanced chat routes for JazzBot

This module handles all chat-related endpoints with improved context management,
token optimization, and streaming responses.
"""

import time
import threading
from flask import Blueprint, request, Response, session, jsonify
from langchain_community.llms import Ollama

from config import Config
from models.memory import ConversationMemory
from services.analyzer import QueryAnalyzer, CategoryDetector
from services.prompt import PromptEngine
from services.retriever import RetrieverService
from services.translator import TranslationService
from utils.helpers import save_response
from utils.logger import logger
from models.memory import thread_local


# Initialize services with enhanced configuration
config = Config()
memory_manager = ConversationMemory(
    max_tokens=getattr(config, 'MEMORY_MAX_TOKENS', 4000),
    max_entries=getattr(config, 'MEMORY_MAX_ENTRIES', 20)
)
query_analyzer = QueryAnalyzer()
category_detector = CategoryDetector()
prompt_engine = PromptEngine(
    max_context_tokens=getattr(config, 'PROMPT_MAX_CONTEXT_TOKENS', 2000)
)
retriever_service = RetrieverService(config)
translator_service = TranslationService()

# Initialize LLM with retry logic
def initialize_llm():
    """Initialize LLM with error handling"""
    try:
        llm = Ollama(
            model=config.LLM_MODEL, 
            temperature=config.LLM_TEMPERATURE,
            timeout=getattr(config, 'LLM_TIMEOUT', 60)
        )
        logger.info(f"LLM initialized successfully with model: {config.LLM_MODEL}")
        return llm
    except Exception as e:
        logger.error(f"Error initializing LLM: {e}")
        return None

llm = initialize_llm()

# Build retriever on startup with status logging
vectorstore, embeddings = retriever_service.build_retriever()
if vectorstore:
    logger.info("Vector store initialized successfully")
else:
    logger.warning("Vector store initialization failed")

# Create blueprint
chat_bp = Blueprint('chat', __name__)


@chat_bp.route("/chat", methods=["POST"])
def chat():
    """Enhanced chat endpoint with advanced context management"""
    try:
        logger.info("=== Chat request received ===")
        data = request.get_json()
        user_input = data.get("message", "").strip()

        if not user_input:
            return Response("Please enter a message.", mimetype="text/plain")

        # Service availability checks
        if vectorstore is None:
            error_response = prompt_engine.build_error_prompt("no_search")
            return Response(error_response, mimetype="text/plain")

        if llm is None:
            error_response = prompt_engine.build_error_prompt("no_llm")
            return Response(error_response, mimetype="text/plain")

        logger.info(f"Processing query: '{user_input[:100]}...'")

        # Initialize enhanced thread local memory
        _initialize_thread_memory()

        # Enhanced query analysis
        analysis_results = _analyze_query(user_input)
        logger.info(f"Query analysis: {analysis_results}")

        # Handle greeting with context awareness
        if analysis_results['is_greeting']:
            response = prompt_engine.build_greeting_response(user_input)
            return Response(
                _stream_response(response, user_input, None), 
                mimetype="text/plain"
            )

        # Enhanced document retrieval
        retrieval_results = _retrieve_documents(user_input, analysis_results)
        
        # Build optimized context
        context_data = _build_context(retrieval_results, analysis_results)
        
        # Generate enhanced prompt
        prompt = _generate_prompt(user_input, analysis_results, context_data)
        
        # Log prompt statistics
        prompt_stats = {
            'length': len(prompt),
            'estimated_tokens': len(prompt) // 4,
            'has_memory': bool(context_data['memory_context']),
            'has_knowledge': bool(context_data['knowledge_context']),
            'categories': analysis_results['detected_categories']
        }
        logger.info(f"Prompt stats: {prompt_stats}")

        # Generate streaming response
        return Response(
            _generate_streaming_response(
                prompt, user_input, analysis_results['primary_category']
            ), 
            mimetype="text/plain"
        )

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        error_response = prompt_engine.build_error_prompt("general")
        return Response(_stream_text(error_response), mimetype="text/plain")


def _initialize_thread_memory():
    """Initialize thread-local memory with session data"""
    try:
        # Get structured memory from session
        session_memory = session.get("chat_memory", [])
        
        # Convert to structured entries if needed
        structured_memory = []
        for entry_data in session_memory:
            try:
                if isinstance(entry_data, dict):
                    # Handle both old and new formats
                    if "timestamp" in entry_data:
                        from models.memory import ConversationEntry
                        structured_memory.append(ConversationEntry.from_dict(entry_data))
                    else:
                        # Legacy format conversion
                        from models.memory import ConversationEntry
                        from datetime import datetime
                        structured_memory.append(ConversationEntry(
                            question=entry_data.get("question", ""),
                            answer=entry_data.get("answer", ""),
                            category=entry_data.get("category"),
                            timestamp=datetime.now(),
                            token_count=memory_manager.token_manager.count_tokens(
                                entry_data.get("question", "") + entry_data.get("answer", "")
                            )
                        ))
            except Exception as e:
                logger.warning(f"Skipping corrupted memory entry: {e}")
                continue
        
        thread_local.chat_memory = structured_memory
        
    except Exception as e:
        logger.error(f"Error initializing thread memory: {e}")
        thread_local.chat_memory = []


def _analyze_query(user_input: str) -> dict:
    """Enhanced query analysis with comprehensive results"""
    try:
        is_greeting = query_analyzer.is_greeting(user_input)
        is_roman_urdu = query_analyzer.is_roman_urdu(user_input)
        last_response = memory_manager.get_last_response()
        is_followup = query_analyzer.is_followup(user_input, last_response)
        detected_categories = category_detector.detect_categories(user_input)
        primary_category = detected_categories[0] if detected_categories else None
        
        # Enhanced analysis
        query_intent = _classify_query_intent(user_input)
        complexity_score = _calculate_query_complexity(user_input)
        
        return {
            'is_greeting': is_greeting,
            'is_roman_urdu': is_roman_urdu,
            'is_followup': is_followup,
            'detected_categories': detected_categories,
            'primary_category': primary_category,
            'last_response': last_response,
            'query_intent': query_intent,
            'complexity_score': complexity_score
        }
    except Exception as e:
        logger.error(f"Error in query analysis: {e}")
        return {
            'is_greeting': False,
            'is_roman_urdu': False,
            'is_followup': False,
            'detected_categories': [],
            'primary_category': None,
            'last_response': "",
            'query_intent': 'general',
            'complexity_score': 1.0
        }


def _classify_query_intent(user_input: str) -> str:
    """Classify the intent of the user query"""
    user_lower = user_input.lower()
    
    if any(word in user_lower for word in ['price', 'cost', 'charges', 'fee']):
        return 'pricing'
    elif any(word in user_lower for word in ['package', 'plan', 'offer']):
        return 'packages'
    elif any(word in user_lower for word in ['detail', 'more', 'explain', 'tell me about']):
        return 'details'
    elif any(word in user_lower for word in ['how to', 'subscribe', 'activate']):
        return 'instructions'
    elif any(word in user_lower for word in ['compare', 'difference', 'vs', 'versus']):
        return 'comparison'
    else:
        return 'general'


def _calculate_query_complexity(user_input: str) -> float:
    """Calculate query complexity score (0.0 to 2.0)"""
    complexity_indicators = {
        'length': len(user_input.split()) / 10,  # Longer queries are more complex
        'questions': user_input.count('?') * 0.3,
        'comparisons': len([w for w in ['vs', 'versus', 'compare', 'difference'] if w in user_input.lower()]) * 0.5,
        'specificity': len([w for w in ['specific', 'detail', 'exact', 'particular'] if w in user_input.lower()]) * 0.4
    }
    
    return min(sum(complexity_indicators.values()), 2.0)


def _retrieve_documents(user_input: str, analysis_results: dict) -> dict:
    """Enhanced document retrieval with smart query optimization"""
    try:
        # Determine search strategy based on analysis
        if analysis_results['is_followup']:
            package_name = query_analyzer.extract_package_name(analysis_results['last_response'])
            if package_name:
                search_query = f"{package_name} {analysis_results['primary_category'] or ''}".strip()
            else:
                search_query = user_input
        else:
            search_query = user_input
        
        # Adjust search parameters based on complexity
        search_k = config.SEARCH_K
        if analysis_results['complexity_score'] > 1.5:
            search_k = int(search_k * 1.5)  # More docs for complex queries
        elif analysis_results['query_intent'] == 'comparison':
            search_k = int(search_k * 2)  # More docs for comparisons
        
        # Search documents
        docs = retriever_service.search_documents(search_query, search_k)
        
        # Filter documents for follow-up queries
        if analysis_results['is_followup'] and 'package_name' in locals():
            docs = [
                doc for doc in docs 
                if package_name.lower() in doc.page_content.lower()
            ]
        
        # Category-specific filtering
        if analysis_results['primary_category']:
            category_docs = [
                doc for doc in docs 
                if doc.metadata.get('category', '').lower() == analysis_results['primary_category'].lower()
            ]
            if category_docs:
                docs = category_docs
        
        logger.info(f"Retrieved {len(docs)} documents for query: '{search_query}'")
        
        return {
            'documents': docs,
            'search_query': search_query,
            'search_k': search_k
        }
        
    except Exception as e:
        logger.error(f"Error in document retrieval: {e}")
        return {
            'documents': [],
            'search_query': user_input,
            'search_k': config.SEARCH_K
        }


def _build_context(retrieval_results: dict, analysis_results: dict) -> dict:
    """Build optimized context from retrieved documents and memory"""
    try:
        # Build knowledge context
        context_parts = []
        for doc in retrieval_results['documents']:
            if doc.page_content.strip():
                metadata_info = (
                    f"[Category: {doc.metadata.get('category', 'unknown')}, "
                    f"Type: {doc.metadata.get('file_type', 'unknown')}]"
                )
                context_parts.append(f"{metadata_info} {doc.page_content}")
        
        knowledge_context = "\n\n".join(context_parts)
        
        # Build memory context based on query type
        if analysis_results['complexity_score'] > 1.0:
            # Use relevant context for complex queries
            memory_context = memory_manager.get_relevant_context(
                retrieval_results['search_query'], 
                max_entries=5
            )
        else:
            # Use recent context for simple queries
            memory_context = memory_manager.get_context_string(
                max_tokens=800,
                include_categories=True
            )
        
        return {
            'knowledge_context': knowledge_context,
            'memory_context': memory_context,
            'context_quality': len(context_parts) / max(retrieval_results['search_k'], 1)
        }
        
    except Exception as e:
        logger.error(f"Error building context: {e}")
        return {
            'knowledge_context': "",
            'memory_context': "",
            'context_quality': 0.0
        }


def _generate_prompt(user_input: str, analysis_results: dict, context_data: dict) -> str:
    """Generate optimized prompt using enhanced prompt engine"""
    try:
        # Extract package name for follow-ups
        package_name = None
        if analysis_results['is_followup'] and analysis_results['last_response']:
            package_name = query_analyzer.extract_package_name(analysis_results['last_response'])
        
        # Generate prompt using enhanced prompt engine
        prompt = prompt_engine.build_context_prompt_main(
            context=context_data['knowledge_context'],
            memory_context=context_data['memory_context'],
            user_input=user_input,
            is_followup=analysis_results['is_followup'],
            package_name=package_name,
            detected_categories=analysis_results['detected_categories'],
            max_response_tokens=_calculate_max_response_tokens(analysis_results)
        )
        
        return prompt
        
    except Exception as e:
        logger.error(f"Error generating prompt: {e}")
        # Fallback to basic prompt
        return f"Please respond to: {user_input}"


def _calculate_max_response_tokens(analysis_results: dict) -> int:
    """Calculate appropriate response length based on query analysis"""
    base_tokens = 150
    
    if analysis_results['query_intent'] == 'details':
        return base_tokens * 2
    elif analysis_results['query_intent'] == 'comparison':
        return base_tokens * 3
    elif analysis_results['complexity_score'] > 1.5:
        return base_tokens * 2
    else:
        return base_tokens


def _generate_streaming_response(prompt: str, user_input: str, primary_category: str):
    """Generate enhanced streaming response with error handling"""
    logger.info("Starting response generation")
    full_response = []
    start_time = time.time()
    
    try:
        # Try streaming first
        try:
            if hasattr(llm, "stream"):
                logger.info("Using streaming response")
                for chunk in llm.stream(prompt):
                    if chunk:
                        full_response.append(chunk)
                        yield chunk
                        time.sleep(0.02)  # Smooth streaming
            else:
                logger.info("Using non-streaming response")
                response = llm.invoke(prompt)
                full_response.append(response)
                
                # Simulate streaming for better UX
                words = response.split()
                for i, word in enumerate(words):
                    if i == 0:
                        yield word
                    else:
                        yield f" {word}"
                    time.sleep(0.05)
                        
        except Exception as stream_error:
            logger.warning(f"Streaming failed, using fallback: {stream_error}")
            response = llm.invoke(prompt)
            full_response.append(response)
            
            # Character-by-character streaming as fallback
            for char in response:
                yield char
                if char in [" ", ".", "!", "?", "\n"]:
                    time.sleep(0.05)
                else:
                    time.sleep(0.01)
    
    except Exception as e:
        logger.error(f"Error in response generation: {e}")
        error_msg = prompt_engine.build_error_prompt("general")
        full_response = [error_msg]
        for char in error_msg:
            yield char
            time.sleep(0.03)
        return
    
    # Process complete response
    complete_response = "".join(full_response).strip()
    generation_time = time.time() - start_time
    
    logger.info(f"Response generated in {generation_time:.2f}s, length: {len(complete_response)}")
    
    # Handle Roman Urdu translation if needed
    if query_analyzer.is_roman_urdu(user_input):
        try:
            roman_urdu_translation = translator_service.translate_to_roman_urdu(complete_response)
            # Optional: yield translation
            # yield f"\n\n[Roman Urdu]: {roman_urdu_translation}"
        except Exception as translation_error:
            logger.error(f"Translation error: {translation_error}")
    
    # Update memory asynchronously
    def update_memory():
        try:
            memory_manager.add_exchange(user_input, complete_response, primary_category)
            save_response(user_input, complete_response, primary_category)
            
            # Log memory stats
            stats = memory_manager.get_memory_stats()
            logger.info(f"Memory updated - Entries: {stats['total_entries']}, Tokens: {stats['total_tokens']}")
            
        except Exception as e:
            logger.error(f"Error updating memory: {e}")
    
    threading.Thread(target=update_memory, daemon=True).start()


def _stream_response(response: str, user_input: str, category: str):
    """Stream a pre-generated response with memory update"""
    words = response.split()
    for i, word in enumerate(words):
        if i == 0:
            yield word
        else:
            yield f" {word}"
        time.sleep(0.05)
    
    # Update memory
    try:
        memory_manager.add_exchange(user_input, response, category)
    except Exception as e:
        logger.error(f"Error updating memory in stream_response: {e}")


def _stream_text(text: str):
    """Stream text character by character"""
    for char in text:
        yield char
        time.sleep(0.03)


@chat_bp.route("/memory/stats", methods=["GET"])
def get_memory_stats():
    """Get detailed memory statistics"""
    try:
        stats = memory_manager.get_memory_stats()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error getting memory stats: {e}")
        return jsonify({"error": "Failed to get memory stats"}), 500


@chat_bp.route("/memory/clear", methods=["POST"])
def clear_memory():
    """Clear conversation memory"""
    try:
        memory_manager.clear_memory()
        logger.info("Memory cleared via API")
        return jsonify({"message": "Memory cleared successfully"})
    except Exception as e:
        logger.error(f"Error clearing memory: {e}")
        return jsonify({"error": "Failed to clear memory"}), 500


@chat_bp.route("/memory/cleanup", methods=["POST"])
def cleanup_memory():
    """Cleanup old memory entries"""
    try:
        data = request.get_json()
        days_to_keep = data.get("days", 7)
        memory_manager.cleanup_old_entries(days_to_keep)
        logger.info(f"Memory cleanup completed - kept {days_to_keep} days")
        return jsonify({"message": f"Memory cleaned - kept last {days_to_keep} days"})
    except Exception as e:
        logger.error(f"Error during memory cleanup: {e}")
        return jsonify({"error": "Failed to cleanup memory"}), 500


@chat_bp.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    try:
        status = {
            "llm_available": llm is not None,
            "vectorstore_available": vectorstore is not None,
            "memory_stats": memory_manager.get_memory_stats(),
            "retriever_status": retriever_service.get_status()
        }
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return jsonify({"error": "Health check failed"}), 500


# Utility functions for other modules
def get_retriever_service():
    """Get retriever service instance"""
    return retriever_service


def get_llm():
    """Get LLM instance"""
    return llm


def get_memory_manager():
    """Get memory manager instance"""
    return memory_manager