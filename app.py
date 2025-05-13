"""
app.py - Flask application for JazzBot, a chatbot for Jazz Telecom (Pakistan).

This script loads documents, creates a FAISS index for retrieval, and provides a chat interface using LangChain and Ollama.
It includes functionality to save and reuse the FAISS index, capture responses in a 'responses' folder, and stores memory of the last 3 question-answer pairs.
"""

from flask import Flask, request, render_template, Response
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import json
import hashlib
import re
from datetime import datetime

app = Flask(__name__)

# Define the path where the FAISS index will be saved
INDEX_PATH = "faiss_index"
INDEX_HASH_PATH = "faiss_index_hash.txt"

# Define the path for the responses folder
RESPONSES_DIR = "responses"

# Initialize chat memory to store the last 3 question-answer pairs
app.chat_memory = []

# Function to clean HTML tags from text
def clean_html_tags(text):
    """Remove HTML tags from text to improve embedding quality."""
    return re.sub(r'<[^>]+>', '', text)

# 1. Load documents with category metadata
def load_documents_with_categories():
    docs = []
    
    # Define your category structure
    categories = {
        "products": ["offers", "data_offers", "packages"],
        "vas": ["offers", "data_offers", "packages"],
        "jazz_rox": ["offers", "data_offers", "packages"],
        "dfs": ["offers", "packages"],
        # Add other categories as needed
    }
    
    # Check if cleaned_data directory exists
    if not os.path.exists("cleaned_data"):
        print("Warning: cleaned_data directory not found. Creating it.")
        os.makedirs("cleaned_data")
        return docs, []  # Return empty docs if no data
    
    # Improved implementation - search in category directories
    file_paths = []
    for category in os.listdir("cleaned_data"):
        category_path = os.path.join("cleaned_data", category)
        
        # Skip if not a directory
        if not os.path.isdir(category_path):
            continue
            
        for file in os.listdir(category_path):
            if file.endswith(".txt"):
                # Determine file type from filename
                file_type = None
                if "offers" in file:
                    file_type = "offers"
                elif "packages" in file:
                    file_type = "packages"
                elif "data_offers" in file:
                    file_type = "data_offers"
                elif "SOP" in file:
                    file_type = "SOP"
                elif "complaints" in file.lower():
                    file_type = "complaints"
                
                file_path = os.path.join(category_path, file)
                print(f"Loading file: {file_path}")
                file_paths.append(file_path)
                
                try:
                    loader = TextLoader(file_path)
                    documents = loader.load()
                    
                    # Add metadata to each document
                    for doc in documents:
                        # Clean HTML tags from the document content
                        doc.page_content = clean_html_tags(doc.page_content)
                        doc.metadata["category"] = category
                        doc.metadata["file_type"] = file_type
                        doc.metadata["filename"] = file
                    
                    docs.extend(documents)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    
    print(f"Loaded {len(docs)} documents in total")
    return docs, file_paths

# Compute a hash of the input files to detect changes
def compute_file_hash(file_paths):
    hasher = hashlib.md5()
    for file_path in sorted(file_paths):
        with open(file_path, "rb") as f:
            hasher.update(f.read())
    return hasher.hexdigest()

# 2. Build retriever with FAISS, BGE embeddings, and metadata filtering
def build_retriever():
    docs, file_paths = load_documents_with_categories()
    
    if not docs:
        print("No documents found. Please ensure your data is loaded correctly.")
        return None, None
        
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    
    print(f"Created {len(chunks)} chunks from documents")
    
    # Ensure metadata is preserved during splitting
    for chunk in chunks:
        if "category" not in chunk.metadata and "filename" in chunk.metadata:
            # Try to infer category from filename if missing
            filename = chunk.metadata["filename"]
            if "products" in filename:
                chunk.metadata["category"] = "products"
            elif "vas" in filename:
                chunk.metadata["category"] = "vas"
            elif "B2C" in filename.lower():
                chunk.metadata["category"] = "B2C"
            elif "jazz_rox" in filename.lower():
                chunk.metadata["category"] = "jazz_rox"
            # Add other category inferences
    
    try:
        # Using HuggingFaceEmbeddings with proper parameters
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Compute hash of the input files
        current_hash = compute_file_hash(file_paths)

        # Check if FAISS index exists and matches the current data
        if os.path.exists(INDEX_PATH) and os.path.exists(INDEX_HASH_PATH):
            with open(INDEX_HASH_PATH, "r") as f:
                saved_hash = f.read().strip()
            if saved_hash == current_hash:
                print(f"Loading existing FAISS index from {INDEX_PATH}...")
                vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
                return vectorstore, embeddings
        
        # If index doesn't exist or data has changed, build a new one
        print("Creating FAISS index from documents...")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        print("FAISS index created successfully")
        
        # Save the FAISS index
        print(f"Saving FAISS index to {INDEX_PATH}...")
        vectorstore.save_local(INDEX_PATH)
        
        # Save the hash of the input files
        with open(INDEX_HASH_PATH, "w") as f:
            f.write(current_hash)
        
        return vectorstore, embeddings
    except Exception as e:
        print(f"Error building vectorstore: {e}")
        return None, None

# Simple keyword-based category detection
def detect_category(query):
    query = query.lower()
    # Define keywords that might indicate specific categories
    category_keywords = {
        "products": ["product", "bundle", "plan"],
        "vas": ["vas", "value added", "service"],
        "jazz_rox": ["rox", "jazz rox", "music"],
        "dfs": ["dfs", "financial", "payment"],
        "digital": ["digital", "app", "online"],
        "rtg": ["rtg", "roaming"],
        "devices": ["device", "handset", "phone", "mobile"],
        "B2C": ["b2c"]  # Added B2C category
    }
    
    # Check for category indicators in the query
    for category, keywords in category_keywords.items():
        for keyword in keywords:
            if keyword in query:
                return category
    
    return None  # No specific category detected

# Initialize resources
try:
    vectorstore, embeddings = build_retriever()
    if vectorstore is None:
        print("WARNING: Vectorstore initialization failed. Search functionality will be limited.")
except Exception as e:
    print(f"Error initializing vectorstore: {e}")
    vectorstore = None
    embeddings = None

try:
    llm = Ollama(model="llama3", temperature=0.7)
    print("LLM initialized successfully")
except Exception as e:
    print(f"Error initializing LLM: {e}")
    llm = None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_input = data.get("message", "").strip()
        if not user_input:
            return Response("Please enter a message.", mimetype='text/plain')
        
        # Check if vectorstore is available
        if vectorstore is None:
            return Response("Search functionality is currently unavailable. Please try again later or contact the administrator.", 
                           mimetype='text/plain')
        
        # Detect if the query is targeting a specific category
        detected_category = detect_category(user_input)
        print(f"Detected category for query '{user_input}': {detected_category}")
        
        # Retrieve relevant documents
        try:
            if detected_category:
                # Filter by detected category if available
                docs = vectorstore.similarity_search(
                    user_input, 
                    k=4,
                    filter={"category": detected_category}
                )
            else:
                # No category filter if none detected
                docs = vectorstore.similarity_search(user_input, k=4)
        except Exception as e:
            print(f"Search error: {e}")
            docs = []
        
        # Log retrieved documents for debugging
        print(f"Retrieved documents: {len(docs)}")
        for i, doc in enumerate(docs):
            print(f"Document {i+1}: [Category: {doc.metadata.get('category', 'unknown')}] {doc.page_content[:100]}...")
        
        # Prepare context with category information
        context_parts = []
        for doc in docs:
            if doc.page_content.strip():
                category = doc.metadata.get("category", "unknown")
                file_type = doc.metadata.get("file_type", "unknown")
                context_parts.append(f"[Category: {category}, Type: {file_type}] {doc.page_content}")
        
        context = "\n\n".join(context_parts)
        print(f"Context length (words): {len(context.split())}")
        print(f"Context: {context[:500]}..." if context else "No context")
        
        # Generate response
        if context and len(context.split()) > 30:
            # Include the last 3 question-answer pairs as memory
            memory_context = "\n".join([f"Previous Q: {item['question']}\nPrevious A: {item['answer']}" for item in app.chat_memory[-3:]])
            prompt = f"""You are JazzBot, a helpful assistant for Jazz Telecom (Pakistan).
You have the following memory of recent interactions:
{memory_context}

You are answering based on the following internal knowledge:
{context}

When referring to offers or packages, always specify which category they belong to (Products, VAS, Jazz ROX, DFS, etc.).
If the user is asking about a specific category like VAS or Jazz ROX, prioritize information from that category.
Use the memory to resolve pronouns (e.g., 'its' refers to the offer or topic from the previous question) and provide contextually relevant answers.
If the price or details are not in the knowledge base, politely state that you need more specific information.

Answer this user query in detail: {user_input}
"""
        else:
            # Include the last 3 question-answer pairs as memory
            memory_context = "\n".join([f"Previous Q: {item['question']}\nPrevious A: {item['answer']}" for item in app.chat_memory[-3:]])
            prompt = f"""You are JazzBot, a helpful assistant for Jazz Telecom (Pakistan).
You have the following memory of recent interactions:
{memory_context}

Answer this user query: {user_input}
If you don't have enough information, politely state that you need more specific details about which Jazz service category you are inquiring about (Products, VAS, Jazz ROX, DFS, etc.).
"""
        
        # Check if LLM is available
        if llm is None:
            return Response("LLM service is currently unavailable. Please try again later.", 
                           mimetype='text/plain')
        
        # Stream response and collect full response
        def generate():
            full_response = ""
            for chunk in llm.stream(prompt):
                full_response += chunk
                yield chunk
            # Update memory and save after streaming is complete
            new_entry = {"question": user_input, "answer": full_response}
            app.chat_memory.append(new_entry)
            if len(app.chat_memory) > 3:
                app.chat_memory = app.chat_memory[-3:]
            
            # Save the response and question
            os.makedirs(RESPONSES_DIR, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            response_file = os.path.join(RESPONSES_DIR, f"response_{timestamp}.txt")
            
            no_answer_keywords = ["don't have enough information", "no data", "not enough details"]
            is_no_answer = any(keyword in full_response.lower() for keyword in no_answer_keywords)
            
            if is_no_answer:
                no_answer_file = os.path.join(RESPONSES_DIR, "no_answer_log.txt")
                with open(no_answer_file, "a", encoding="utf-8") as f:
                    f.write(f"Question: {user_input}\n")
                    f.write(f"Response: {full_response}\n")
                    f.write("-" * 50 + "\n")
            else:
                with open(response_file, "w", encoding="utf-8") as f:
                    f.write(f"Question: {user_input}\n")
                    f.write(f"Response: {full_response}\n")
        
        return Response(generate(), mimetype='text/plain')
    
    except Exception as e:
        return Response(f"Error: {str(e)}", mimetype='text/plain')

@app.route("/admin/upload", methods=["POST"])
def admin_upload():
    """Endpoint for admins to upload new category data"""
    global vectorstore, embeddings
    try:
        if 'file' not in request.files:
            return Response("No file part", status=400)
            
        file = request.files['file']
        category = request.form.get('category', 'unknown')
        file_type = request.form.get('file_type', 'unknown')
        
        if file.filename == '':
            return Response("No selected file", status=400)
            
        # Save file to appropriate directory
        category_dir = os.path.join("cleaned_data", category)
        os.makedirs(category_dir, exist_ok=True)
        
        filename = f"{category}_{file_type}.txt"
        file_path = os.path.join(category_dir, filename)
        file.save(file_path)
        
        # Delete the existing FAISS index to force a rebuild
        if os.path.exists(INDEX_PATH):
            import shutil
            shutil.rmtree(INDEX_PATH)
            print(f"Deleted existing FAISS index at {INDEX_PATH}")
        if os.path.exists(INDEX_HASH_PATH):
            os.remove(INDEX_HASH_PATH)
            print(f"Deleted existing hash file at {INDEX_HASH_PATH}")
        
        # Rebuild the retriever with the new data
        try:
            vectorstore, embeddings = build_retriever()
            if vectorstore is None:
                return Response("File uploaded but failed to rebuild index.", status=500)
            return Response("File uploaded successfully and index rebuilt", status=200)
        except Exception as e:
            return Response(f"File uploaded but error rebuilding index: {str(e)}", status=500)
    
    except Exception as e:
        return Response(f"Error: {str(e)}", status=500)

@app.route("/status")
def status():
    """Simple status endpoint to check if services are running"""
    return {
        "status": "online",
        "vectorstore": "available" if vectorstore is not None else "unavailable",
        "llm": "available" if llm is not None else "unavailable"
    }

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=6068)