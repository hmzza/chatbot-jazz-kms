"""
app.py - Enhanced Flask application for JazzBot, a chatbot for Jazz Telecom (Pakistan).

This improved version fixes critical issues:
- Fixed session context error in streaming responses
- Updated deprecated LangChain imports
- Fixed follow-up detection and conversation flow
- Improved memory management
- Enhanced error handling
- Fixed langchain_ollama import issue
- FIXED: Proper streaming response with typing effect
"""

from flask import Flask, request, render_template, Response, session, g

# Updated import to use the correct LangChain community package
from langchain_community.llms import Ollama  # Fixed import
from langchain_community.embeddings import HuggingFaceEmbeddings  # Updated import
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import json
import hashlib
import re
from datetime import datetime
import time
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("jazzbot.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get(
    "SECRET_KEY", "your_secret_key_here_change_in_production"
)

# Thread-local storage for request context
thread_local = threading.local()


# Configuration
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


class ConversationMemory:
    """Thread-safe conversation memory management"""

    def __init__(self, max_size: int = config.MEMORY_SIZE):
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
        session["chat_memory"] = []
        if hasattr(thread_local, "chat_memory"):
            delattr(thread_local, "chat_memory")


class DocumentProcessor:
    """Enhanced document processing and retrieval"""

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", "", text)
        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)
        # Remove special characters but keep important ones
        text = re.sub(r"[^\w\s\-\+\.\,\:\;\(\)\[\]\/\&\%\$\#\@\!]", " ", text)
        return text.strip()

    @staticmethod
    def load_documents_with_categories() -> Tuple[List, List[str]]:
        """Load documents with enhanced metadata"""
        docs = []
        file_paths = []

        if not os.path.exists(config.DATA_DIR):
            os.makedirs(config.DATA_DIR)
            logger.warning(f"Created {config.DATA_DIR} directory - no documents found")
            return docs, file_paths

        for category in os.listdir(config.DATA_DIR):
            category_path = os.path.join(config.DATA_DIR, category)
            if not os.path.isdir(category_path):
                continue

            for file in os.listdir(category_path):
                if not file.endswith(".txt"):
                    continue

                file_path = os.path.join(category_path, file)
                file_paths.append(file_path)

                # Determine file type with better logic
                file_type = "unknown"
                file_lower = file.lower()
                if "offer" in file_lower:
                    file_type = "offers"
                elif "package" in file_lower:
                    file_type = "packages"
                elif "data" in file_lower:
                    file_type = "data_offers"
                elif "bundle" in file_lower:
                    file_type = "bundles"

                try:
                    loader = TextLoader(file_path, encoding="utf-8")
                    documents = loader.load()

                    for doc in documents:
                        doc.page_content = DocumentProcessor.clean_text(
                            doc.page_content
                        )
                        doc.metadata.update(
                            {
                                "category": category,
                                "file_type": file_type,
                                "filename": file,
                                "source_path": file_path,
                            }
                        )

                    docs.extend(documents)
                    logger.info(f"Loaded {len(documents)} documents from {file_path}")

                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")

        logger.info(f"Total documents loaded: {len(docs)}")
        return docs, file_paths


class CategoryDetector:
    """Enhanced category detection"""

    CATEGORY_KEYWORDS = {
        "B2C": ["b2c", "consumer", "individual", "personal"],
        "B2B": ["b2b", "business", "corporate", "enterprise"],
        "prepaid": ["prepaid", "pre-paid", "recharge", "top-up"],
        "postpaid": ["postpaid", "post-paid", "monthly", "contract"],
        "data": ["data", "internet", "mb", "gb", "4g", "5g"],
        "voice": ["call", "calls", "minutes", "voice", "talk"],
        "sms": ["sms", "text", "message", "messaging"],
        "facebook": ["facebook", "fb"],
        "youtube": ["youtube", "yt"],
        "whatsapp": ["whatsapp", "wa"],
    }

    @classmethod
    def detect_categories(cls, query: str) -> List[str]:
        """Detect multiple categories from query"""
        query_lower = query.lower()
        detected = []

        for category, keywords in cls.CATEGORY_KEYWORDS.items():
            if any(keyword in query_lower for keyword in keywords):
                detected.append(category)

        return detected

    @classmethod
    def detect_primary_category(cls, query: str) -> Optional[str]:
        """Detect primary category"""
        categories = cls.detect_categories(query)
        return categories[0] if categories else None


class QueryAnalyzer:
    """Enhanced query analysis"""

    GREETING_PATTERNS = [
        r"^(hi|hello|hey|salam|assalam|good\s+(morning|afternoon|evening))$",
        r"^(how\s+are\s+you|what\'s\s+up)$",
    ]

    FOLLOWUP_PATTERNS = [
        r"^(yes|yeah|yep|sure|ok|okay)$",
        r"^(show\s+me\s+)?(more\s+)?(details?|info)",
        r"^(tell\s+me\s+more|give\s+me\s+more)",
        r"^(what\s+about|how\s+about)",
        r"^(can\s+you\s+explain|explain\s+more)",
    ]

    PACKAGE_EXTRACTION_PATTERNS = [
        r"(?:we have|offers?|try)\s+(?:a\s+|the\s+)?([^.]+?)(?:\s+(?:offer|bundle|package|plan))",
        r"(facebook|youtube|whatsapp|instagram)\s+(?:premium\s+|daily\s+|weekly\s+|monthly\s+)?(?:offer|bundle|package)",
        r"(?:jazz\s+)?(super\s+card|smart\s+bundle)",
    ]

    @classmethod
    def is_greeting(cls, query: str) -> bool:
        """Check if query is a greeting"""
        query_clean = query.strip().lower()
        return any(
            re.match(pattern, query_clean, re.IGNORECASE)
            for pattern in cls.GREETING_PATTERNS
        )

    @classmethod
    def is_followup(cls, query: str, last_response: str = "") -> bool:
        """Check if query is a follow-up question"""
        query_clean = query.strip().lower()
        is_followup_pattern = any(
            re.match(pattern, query_clean, re.IGNORECASE)
            for pattern in cls.FOLLOWUP_PATTERNS
        )
        has_details_offer = any(
            phrase in last_response.lower()
            for phrase in [
                "would you like more details",
                "want more details",
                "details?",
                "more info",
            ]
        )
        return is_followup_pattern and has_details_offer

    @classmethod
    def extract_package_name(cls, text: str) -> Optional[str]:
        """Extract package name from text"""
        for pattern in cls.PACKAGE_EXTRACTION_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None


class PromptEngine:
    """Enhanced prompt generation"""

    @staticmethod
    def build_context_prompt(
        context: str,
        memory_context: str,
        user_input: str,
        is_followup: bool = False,
        package_name: str = None,
        detected_categories: List[str] = None,
    ) -> str:
        """Build enhanced context-aware prompt"""

        base_instructions = """You are JazzBot, a helpful assistant for Jazz Telecom Pakistan. 

CORE PRINCIPLES:
- Provide accurate information based on the provided knowledge
- Be conversational and friendly
- Keep responses concise and to the point
- Always specify the category when mentioning packages (B2C, B2B, etc.)
- Use bullet points for detailed information
- Don't make up information if not in the knowledge base"""

        if is_followup and package_name:
            return f"""{base_instructions}

CONVERSATION HISTORY:
{memory_context}

KNOWLEDGE BASE:
{context}

TASK: You are being asked for more details about "{package_name}". Provide comprehensive information in bullet point format.

REQUIRED FORMAT:
- Package Name: [Name]
- Category: [B2C/B2B/etc.]
- Price: [Amount]
- Validity: [Duration]
- Data/Internet: [Amount if applicable]
- Calls: [Details if applicable]
- SMS: [Details if applicable]
- Subscription Code: [Code if available]
- Other Details: [Any additional info]

If any information is not available, write "Not specified in available data"

USER REQUEST: {user_input}"""

        elif context and len(context.split()) > 15:
            category_info = (
                f"Detected categories: {', '.join(detected_categories)}"
                if detected_categories
                else ""
            )

            return f"""{base_instructions}

CONVERSATION HISTORY:
{memory_context}

KNOWLEDGE BASE:
{context}

{category_info}

INSTRUCTIONS:
- Answer based on the knowledge provided above
- Keep response under 100 words
- Specify category when mentioning packages (B2C, B2B, etc.)
- If you have detailed info available, end with "Would you like more details?"
- Be specific about what's available
- For user input like AOA or aoa or Assalam o alaikum: "Walaikum Salam! I'm JazzBot, your Jazz Telecom assistant. How can I help you today?"

USER REQUEST: {user_input}"""

        else:
            return f"""{base_instructions}

CONVERSATION HISTORY:
{memory_context}

SITUATION: No specific knowledge available for this query.

INSTRUCTIONS:
- For greetings: "Hi! I'm JazzBot, your Jazz Telecom assistant. How can I help you today?"
- For specific queries: "I'd be happy to help! Could you please specify what type of Jazz service you're looking for? For example: B2C packages, B2B solutions, data offers, voice plans, or SMS bundles?"
- Keep response under 50 words
- Be helpful and guide the user to provide more specific information
- For user input like AOA or aoa or Assalam o alaikum: "Walaikum Salam! I'm JazzBot, your Jazz Telecom assistant. How can I help you today?"

USER REQUEST: {user_input}"""


# Initialize global components
memory_manager = ConversationMemory()
category_detector = CategoryDetector()
query_analyzer = QueryAnalyzer()
prompt_engine = PromptEngine()


def compute_file_hash(file_paths: List[str]) -> str:
    """Compute hash of input files"""
    hasher = hashlib.md5()
    for file_path in sorted(file_paths):
        try:
            with open(file_path, "rb") as f:
                hasher.update(f.read())
        except Exception as e:
            logger.error(f"Error hashing {file_path}: {e}")
    return hasher.hexdigest()


def build_retriever() -> Tuple[Optional[object], Optional[object]]:
    """Build enhanced FAISS retriever"""
    try:
        docs, file_paths = DocumentProcessor.load_documents_with_categories()

        if not docs:
            logger.warning("No documents found for indexing")
            return None, None

        # Enhanced text splitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        )

        chunks = splitter.split_documents(docs)
        logger.info(f"Created {len(chunks)} chunks from {len(docs)} documents")

        # Initialize embeddings with correct import
        embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # Check if we can reuse existing index
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
                        allow_dangerous_deserialization=True,
                    )
                    return vectorstore, embeddings
            except Exception as e:
                logger.error(f"Error loading existing index: {e}")

        # Build new index
        logger.info("Building new FAISS index...")
        vectorstore = FAISS.from_documents(chunks, embeddings)

        # Save index and hash
        vectorstore.save_local(config.INDEX_PATH)
        with open(config.INDEX_HASH_PATH, "w") as f:
            f.write(current_hash)

        logger.info("FAISS index built and saved successfully")
        return vectorstore, embeddings

    except Exception as e:
        logger.error(f"Error building retriever: {e}")
        return None, None


def save_response(question: str, response: str, category: str = None):
    """Save response to file"""
    try:
        os.makedirs(config.RESPONSES_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        response_file = os.path.join(config.RESPONSES_DIR, f"response_{timestamp}.txt")

        with open(response_file, "w", encoding="utf-8") as f:
            f.write(f"Question: {question}\n")
            f.write(f"Response: {response}\n")
            f.write(f"Category: {category or 'Unknown'}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")

    except Exception as e:
        logger.error(f"Error saving response: {e}")


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
    # Fixed LLM initialization using correct import
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

        # Copy session data to thread local before streaming
        try:
            if "chat_memory" in session:
                thread_local.chat_memory = session["chat_memory"].copy()
            else:
                thread_local.chat_memory = []
        except:
            thread_local.chat_memory = []

        # Analyze query
        is_greeting = query_analyzer.is_greeting(user_input)
        last_response = memory_manager.get_last_response()
        is_followup = query_analyzer.is_followup(user_input, last_response)
        detected_categories = category_detector.detect_categories(user_input)
        primary_category = detected_categories[0] if detected_categories else None

        logger.info(
            f"Query analysis - Greeting: {is_greeting}, Follow-up: {is_followup}, Categories: {detected_categories}"
        )

        # Handle greeting with streaming effect
        if is_greeting:
            response = "Hi! I'm JazzBot, your Jazz Telecom assistant. How can I help you today?"

            def stream_greeting():
                words = response.split()
                for i, word in enumerate(words):
                    if i == 0:
                        yield word
                    else:
                        yield f" {word}"
                    time.sleep(0.05)  # Small delay for typing effect
                # Update memory after streaming
                memory_manager.add_exchange(user_input, response, None)

            return Response(stream_greeting(), mimetype="text/plain")

        # Document retrieval
        docs = []
        package_name = None

        try:
            if is_followup:
                package_name = query_analyzer.extract_package_name(last_response)
                search_query = (
                    f"{package_name} {primary_category or ''}".strip()
                    if package_name
                    else user_input
                )
                logger.info(f"Follow-up search for package: {package_name}")
            else:
                search_query = user_input

            # Enhanced search
            docs = vectorstore.similarity_search(search_query, k=config.SEARCH_K)

            # Filter out discontinued offers
            docs = [
                doc for doc in docs if "discontinued" not in doc.page_content.lower()
            ]

            # For follow-ups, ensure package name match
            if is_followup and package_name:
                docs = [
                    doc
                    for doc in docs
                    if package_name.lower() in doc.page_content.lower()
                ]

            logger.info(f"Retrieved {len(docs)} relevant documents")

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
        prompt = prompt_engine.build_context_prompt(
            context=context,
            memory_context=memory_context,
            user_input=user_input,
            is_followup=is_followup,
            package_name=package_name,
            detected_categories=detected_categories,
        )

        logger.info(f"Generated prompt length: {len(prompt)} characters")

        # FIXED: Stream response with proper word-by-word streaming
        def generate():
            full_response = []
            start_time = time.time()

            try:
                # Try to use streaming if available
                try:
                    # Check if the LLM supports streaming
                    if hasattr(llm, "stream"):
                        logger.info("Using LLM streaming")
                        for chunk in llm.stream(prompt):
                            if chunk:
                                full_response.append(chunk)
                                yield chunk
                                time.sleep(0.02)  # Small delay for better typing effect
                    else:
                        # Fallback: Get full response and simulate streaming
                        logger.info(
                            "Simulating streaming (LLM doesn't support native streaming)"
                        )
                        response = llm.invoke(prompt)
                        full_response.append(response)

                        # Stream word by word for typing effect
                        words = response.split()
                        for i, word in enumerate(words):
                            if i == 0:
                                yield word
                            else:
                                yield f" {word}"
                            time.sleep(0.08)  # Typing effect delay

                except Exception as stream_error:
                    logger.warning(f"Streaming failed, using invoke: {stream_error}")
                    # Final fallback: use invoke and simulate streaming
                    response = llm.invoke(prompt)
                    full_response.append(response)

                    # Stream character by character for better typing effect
                    for char in response:
                        yield char
                        if char in [" ", ".", "!", "?", "\n"]:
                            time.sleep(0.05)  # Pause at word/sentence boundaries
                        else:
                            time.sleep(0.01)  # Regular typing speed

            except Exception as e:
                logger.error(f"Error in generation: {e}")
                error_msg = "I apologize, but I encountered an error processing your request. Please try again."
                full_response = [error_msg]

                # Stream error message
                for char in error_msg:
                    yield char
                    time.sleep(0.03)
                return

            # Post-processing - this happens after streaming completes
            complete_response = "".join(full_response).strip()

            # Update memory in a separate thread to avoid context issues
            def update_memory():
                try:
                    # Use Flask's test request context for memory update
                    with app.test_request_context():
                        # Simulate session for memory update
                        from flask import session as test_session

                        if hasattr(thread_local, "chat_memory"):
                            test_session["chat_memory"] = thread_local.chat_memory

                        memory_manager.add_exchange(
                            user_input, complete_response, primary_category
                        )
                        save_response(user_input, complete_response, primary_category)
                        logger.info(
                            f"Generated response length: {len(complete_response)} characters"
                        )
                except Exception as e:
                    logger.error(f"Error updating memory: {e}")

            # Run memory update in background
            threading.Thread(target=update_memory, daemon=True).start()

        return Response(generate(), mimetype="text/plain")

    except Exception as e:
        logger.error(f"Chat error: {e}")
        # Stream error message for consistency
        error_msg = (
            "I apologize, but I encountered an unexpected error. Please try again."
        )

        def stream_error():
            for char in error_msg:
                yield char
                time.sleep(0.03)

        return Response(stream_error(), mimetype="text/plain")


@app.route("/admin/upload", methods=["POST"])
def admin_upload():
    """Enhanced admin upload endpoint"""
    global vectorstore, embeddings

    try:
        if "file" not in request.files:
            return Response("No file provided", status=400)

        file = request.files["file"]
        category = request.form.get("category", "unknown")
        file_type = request.form.get("file_type", "unknown")

        if file.filename == "":
            return Response("No file selected", status=400)

        # Save file
        category_dir = os.path.join(config.DATA_DIR, category)
        os.makedirs(category_dir, exist_ok=True)

        filename = (
            f"{category}_{file_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        file_path = os.path.join(category_dir, filename)
        file.save(file_path)

        logger.info(f"File uploaded: {file_path}")

        # Rebuild index
        if os.path.exists(config.INDEX_PATH):
            import shutil

            shutil.rmtree(config.INDEX_PATH)

        if os.path.exists(config.INDEX_HASH_PATH):
            os.remove(config.INDEX_HASH_PATH)

        vectorstore, embeddings = build_retriever()

        success_msg = (
            "File uploaded and index rebuilt successfully"
            if vectorstore
            else "File uploaded but index rebuild failed"
        )
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
            "search_k": config.SEARCH_K,
        },
        "timestamp": datetime.now().isoformat(),
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
    app.run(debug=True, host="0.0.0.0", port=6069)
