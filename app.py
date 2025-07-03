"""
app.py - Main Flask application for JazzBot
"""

from flask import Flask, session
import os
import threading
from datetime import datetime

from config import Config
from utils.logger import setup_logger
from models.memory import ConversationMemory

# from services.retriever import build_retriever
from services.retriever import RetrieverService
from services.analyzer import QueryAnalyzer
from routes.chat import chat_bp
from routes.admin import admin_bp

# Configure logging
logger = setup_logger(__name__, "jazzbot.log")

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get(
    "SECRET_KEY", "your_secret_key_here_change_in_production"
)

# Register blueprints
app.register_blueprint(chat_bp)
app.register_blueprint(admin_bp, url_prefix="/admin")

# Thread-local storage for request context
thread_local = threading.local()

# Global components
config = Config()
memory_manager = ConversationMemory()
query_analyzer = QueryAnalyzer()

# Initialize resources
logger.info("Initializing JazzBot...")

try:
    config = Config()

    # Create an instance of RetrieverService
    retriever_service = RetrieverService(config)

    # Call the build_retriever method
    vectorstore, embeddings = retriever_service.build_retriever()

    if vectorstore is None:
        logger.error("Vectorstore initialization failed")
    else:
        logger.info("Vectorstore initialized successfully")
except Exception as e:
    logger.error(f"Error initializing vectorstore: {e}")
    vectorstore = None
    embeddings = None

# Make global components available to blueprints
app.config["vectorstore"] = vectorstore
app.config["embeddings"] = embeddings
app.config["memory_manager"] = memory_manager
app.config["query_analyzer"] = query_analyzer
app.config["jazzbot_config"] = config
app.config["thread_local"] = thread_local


@app.before_request
def before_request():
    """Store request context in thread local"""
    thread_local.request_context = True


@app.route("/")
def index():
    """Serve main page"""
    from flask import render_template

    return render_template("index.html")


@app.route("/status")
def status():
    """Enhanced status endpoint"""
    memory_count = len(memory_manager.get_memory())

    return {
        "status": "online",
        "vectorstore": "available" if vectorstore else "unavailable",
        "memory_size": memory_count,
        "config": {
            "memory_limit": config.MEMORY_SIZE,
            "chunk_size": config.CHUNK_SIZE,
            "search_k": config.SEARCH_K,
        },
        "timestamp": datetime.now().isoformat(),
    }


@app.errorhandler(500)
def internal_error(error):
    """Handle internal errors"""
    logger.error(f"Internal server error: {error}")
    from flask import Response

    return Response("Internal server error occurred", status=500)


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    from flask import Response

    return Response("Page not found", status=404)


if __name__ == "__main__":
    logger.info("Starting JazzBot Flask application...")
    app.run(debug=True, host="0.0.0.0", port=6062)
