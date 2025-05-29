"""
app.py - Flask application for JazzBot, a chatbot for Jazz Telecom (Pakistan).

This file contains the main application logic, routes, and initialization.
Utility functions and configurations are imported from separate modules.
"""

from flask import Flask, request, render_template, Response, session
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import time
import threading
from datetime import datetime
import logging

# Import utilities and configurations
from config import config
from utils import (
    ConversationMemory, DocumentProcessor, CategoryDetector, QueryAnalyzer,
    PromptEngine, compute_file_hash, save_response, logger, thread_local
)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your_secret_key_here_change_in_production')

# Initialize components
memory_manager = ConversationMemory(max_size=config.MEMORY_SIZE)
category_detector = CategoryDetector()
query_analyzer = QueryAnalyzer()
prompt_engine = PromptEngine()

def build_retriever():
    """Build enhanced FAISS retriever"""
    try:
        docs, file_paths = DocumentProcessor.load_documents_with_categories(config.DATA_DIR)
        
        if not docs:
            logger.warning("No documents found for indexing")
            return None, None
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
        
        chunks = splitter.split_documents(docs)
        logger.info(f"Created {len(chunks)} chunks from {len(docs)} documents")
        
        embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        current_hash = compute_file_hash(file_paths)
        
        if os.path.exists(config.INDEX_PATH) and os.path.exists(config.INDEX_HASH_PATH):
            try:
                with open(config.INDEX_HASH_PATH, "r") as f:
                    saved_hash = f.read().strip()
                
                if saved_hash == current_hash:
                    logger.info("Reusing existing FAISS index")
                    vectorstore = FAISS.load_local(
                        config.INDEX_PATH, 
                        embeddings, 
                        allow_dangerous_deserialization=True
                    )
                    return vectorstore, embeddings
            except Exception as e:
                logger.error(f"Error loading existing index: {e}")
        
        logger.info("Building new FAISS index...")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        
        vectorstore.save_local(config.INDEX_PATH)
        with open(config.INDEX_HASH_PATH, "w") as f:
            f.write(current_hash)
        
        logger.info("FAISS index built and saved successfully")
        return vectorstore, embeddings
        
    except Exception as e:
        logger.error(f"Error building retriever: {e}")
        return None, None

# Initialize resources
logger.info("Initializing JazzBot...")

try:
    vectorstore, embeddings = build_retriever()
    if vectorstore is None:
        logger.error("Vectorstore initialization failed")
    else:
        logger.info("Vectorstore initialized successfully")
except Exception as e:
    logger.error(f"Error initializing vectorstore: {e}")
    vectorstore = None
    embeddings = None

try:
    llm = Ollama(model=config.LLM_MODEL, temperature=config.LLM_TEMPERATURE)
    logger.info("LLM initialized successfully")
except Exception as e:
    logger.error(f"Error initializing LLM: {e}")
    llm = None

@app.before_request
def before_request():
    """Store request context in thread local"""
    thread_local.request_context = True

@app.route("/")
def index():
    """Serve main page"""
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    """Enhanced chat endpoint with FIXED streaming for typing effect"""
    try:
        data = request.get_json()
        user_input = data.get("message", "").strip()
        
        if not user_input:
            return Response("Please enter a message.", mimetype='text/plain')
        
        if vectorstore is None:
            return Response("Search functionality is currently unavailable. Please try again later.", mimetype='text/plain')
        
        if llm is None:
            return Response("AI service is currently unavailable. Please try again later.", mimetype='text/plain')
        
        logger.info(f"Processing query: {user_input}")
        
        try:
            if 'chat_memory' in session:
                thread_local.chat_memory = session['chat_memory'].copy()
            else:
                thread_local.chat_memory = []
        except:
            thread_local.chat_memory = []
        
        is_greeting = query_analyzer.is_greeting(user_input)
        last_response = memory_manager.get_last_response()
        is_followup = query_analyzer.is_followup(user_input, last_response)
        detected_categories = category_detector.detect_categories(user_input)
        primary_category = detected_categories[0] if detected_categories else None
        
        logger.info(f"Query analysis - Greeting: {is_greeting}, Follow-up: {is_followup}, Categories: {detected_categories}")
        
        if is_greeting:
            response = "Hi! I'm JazzBot, your Jazz Telecom assistant. How can I help you today?"
            
            def stream_greeting():
                words = response.split()
                for i, word in enumerate(words):
                    if i == 0:
                        yield word
                    else:
                        yield f" {word}"
                    time.sleep(0.05)
                memory_manager.add_exchange(user_input, response, None)
            
            return Response(stream_greeting(), mimetype='text/plain')
        
        docs = []
        package_name = None
        
        try:
            if is_followup:
                package_name = query_analyzer.extract_package_name(last_response)
                search_query = f"{package_name} {primary_category or ''}".strip() if package_name else user_input
                logger.info(f"Follow-up search for package: {package_name}")
            else:
                search_query = user_input
            
            # Increase k to retrieve more documents for comprehensive details
            docs = vectorstore.similarity_search(search_query, k=10)
            docs = [doc for doc in docs if "discontinued" not in doc.page_content.lower()]
            
            if is_followup and package_name:
                docs = [doc for doc in docs if package_name.lower() in doc.page_content.lower()]
            
            logger.info(f"Retrieved {len(docs)} relevant documents")
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            docs = []
        
        context_parts = []
        for doc in docs:
            if doc.page_content.strip():
                metadata_info = f"[Category: {doc.metadata.get('category', 'unknown')}, Type: {doc.metadata.get('file_type', 'unknown')}]"
                context_parts.append(f"{metadata_info} {doc.page_content}")
        
        context = "\n\n".join(context_parts)
        memory_context = memory_manager.get_context_string()
        
        prompt = prompt_engine.build_context_prompt(
            context=context,
            memory_context=memory_context,
            user_input=user_input,
            is_followup=is_followup,
            package_name=package_name,
            detected_categories=detected_categories
        )
        
        logger.info(f"Generated prompt length: {len(prompt)} characters")
        
        def generate():
            full_response = []
            start_time = time.time()
            
            try:
                try:
                    if hasattr(llm, 'stream'):
                        logger.info("Using LLM streaming")
                        for chunk in llm.stream(prompt):
                            if chunk:
                                full_response.append(chunk)
                                yield chunk
                                time.sleep(0.02)
                    else:
                        logger.info("Simulating streaming (LLM doesn't support native streaming)")
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
                        if char in [' ', '.', '!', '?', '\n']:
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
            
            def update_memory():
                try:
                    with app.test_request_context():
                        from flask import session as test_session
                        if hasattr(thread_local, 'chat_memory'):
                            test_session['chat_memory'] = thread_local.chat_memory
                        memory_manager.add_exchange(user_input, complete_response, primary_category)
                        save_response(user_input, complete_response, primary_category, config.RESPONSES_DIR)
                        logger.info(f"Generated response length: {len(complete_response)} characters")
                except Exception as e:
                    logger.error(f"Error updating memory: {e}")
            
            threading.Thread(target=update_memory, daemon=True).start()
        
        return Response(generate(), mimetype='text/plain')
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        error_msg = "I apologize, but I encountered an unexpected error. Please try again."
        
        def stream_error():
            for char in error_msg:
                yield char
                time.sleep(0.03)
        
        return Response(stream_error(), mimetype='text/plain')

@app.route("/admin/upload", methods=["POST"])
def admin_upload():
    """Enhanced admin upload endpoint"""
    global vectorstore, embeddings
    
    try:
        if 'file' not in request.files:
            return Response("No file provided", status=400)
        
        file = request.files['file']
        category = request.form.get('category', 'unknown')
        file_type = request.form.get('file_type', 'unknown')
        
        if file.filename == '':
            return Response("No file selected", status=400)
        
        category_dir = os.path.join(config.DATA_DIR, category)
        os.makedirs(category_dir, exist_ok=True)
        
        filename = f"{category}_{file_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        file_path = os.path.join(category_dir, filename)
        file.save(file_path)
        
        logger.info(f"File uploaded: {file_path}")
        
        if os.path.exists(config.INDEX_PATH):
            import shutil
            shutil.rmtree(config.INDEX_PATH)
        
        if os.path.exists(config.INDEX_HASH_PATH):
            os.remove(config.INDEX_HASH_PATH)
        
        vectorstore, embeddings = build_retriever()
        
        success_msg = "File uploaded and index rebuilt successfully" if vectorstore else "File uploaded but index rebuild failed"
        logger.info(success_msg)
        
        return Response(success_msg, status=200)
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return Response(f"Upload failed: {str(e)}", status=500)

@app.route("/admin/clear_memory", methods=["POST"])
def clear_memory():
    """Clear conversation memory"""
    try:
        memory_manager.clear_memory()
        return Response("Memory cleared successfully", status=200)
    except Exception as e:
        logger.error(f"Error clearing memory: {e}")
        return Response("Failed to clear memory", status=500)

@app.route("/status")
def status():
    """Enhanced status endpoint"""
    memory_count = len(memory_manager.get_memory())
    
    return {
        "status": "online",
        "vectorstore": "available" if vectorstore else "unavailable",
        "llm": "available" if llm else "unavailable",
        "memory_size": memory_count,
        "config": {
            "memory_limit": config.MEMORY_SIZE,
            "chunk_size": config.CHUNK_SIZE,
            "search_k": config.SEARCH_K
        },
        "timestamp": datetime.now().isoformat()
    }

@app.errorhandler(500)
def internal_error(error):
    """Handle internal errors"""
    logger.error(f"Internal server error: {error}")
    return Response("Internal server error occurred", status=500)

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return Response("Page not found", status=404)

if __name__ == "__main__":
    logger.info("Starting JazzBot Flask application...")
    app.run(debug=True, host="0.0.0.0", port=6064)