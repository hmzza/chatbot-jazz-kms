from flask import Flask, request, render_template, Response
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings  # Original import
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import json

app = Flask(__name__)

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
        return docs  # Return empty docs if no data
    
    # Improved implementation - search in category directories
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
                
                file_path = os.path.join(category_path, file)
                print(f"Loading file: {file_path}")
                
                try:
                    loader = TextLoader(file_path)
                    documents = loader.load()
                    
                    # Add metadata to each document
                    for doc in documents:
                        doc.metadata["category"] = category
                        doc.metadata["file_type"] = file_type
                        doc.metadata["filename"] = file
                    
                    docs.extend(documents)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    
    print(f"Loaded {len(docs)} documents in total")
    return docs

# 2. Build retriever with FAISS, BGE embeddings, and metadata filtering
def build_retriever():
    docs = load_documents_with_categories()
    
    if not docs:
        print("No documents found. Please ensure your data is loaded correctly.")
        return None
        
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
            # Add other category inferences
    
    try:
        # Using HuggingFaceEmbeddings with proper parameters
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",  # Using a different model that's more reliable
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        print("Creating FAISS index from documents...")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        print("FAISS index created successfully")
        
        # Return the vectorstore directly instead of as_retriever for more flexibility
        return vectorstore
    except Exception as e:
        print(f"Error building vectorstore: {e}")
        return None

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
        "devices": ["device", "handset", "phone", "mobile"]
    }
    
    # Check for category indicators in the query
    for category, keywords in category_keywords.items():
        for keyword in keywords:
            if keyword in query:
                return category
    
    return None  # No specific category detected

# Initialize resources
try:
    vectorstore = build_retriever()
    if vectorstore is None:
        print("WARNING: Vectorstore initialization failed. Search functionality will be limited.")
except Exception as e:
    print(f"Error initializing vectorstore: {e}")
    vectorstore = None

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
        
        # Prepare context with category information
        context_parts = []
        for doc in docs:
            if doc.page_content.strip():
                category = doc.metadata.get("category", "unknown")
                file_type = doc.metadata.get("file_type", "unknown")
                context_parts.append(f"[Category: {category}, Type: {file_type}] {doc.page_content}")
        
        context = "\n\n".join(context_parts)
        
        # If relevant context exists, use RAG; otherwise, fallback to plain chatbot
        if context and len(context.split()) > 30:
            prompt = f"""You are JazzBot, a helpful assistant for Jazz Telecom (Pakistan).
You are answering based on the following internal knowledge:

{context}

When referring to offers or packages, always specify which category they belong to (Products, VAS, Jazz ROX, DFS, etc.)
If the user is asking about a specific category like VAS or Jazz ROX, prioritize information from that category.

Answer this user query in detail: {user_input}
"""
        else:
            prompt = f"""You are JazzBot, a helpful assistant for Jazz Telecom (Pakistan). 
Answer this user query: {user_input}
If you don't have enough information, politely state that you need more specific details about which Jazz service category the user is inquiring about (Products, VAS, Jazz ROX, DFS, etc.).
"""
        
        # Check if LLM is available
        if llm is None:
            return Response("LLM service is currently unavailable. Please try again later.", 
                           mimetype='text/plain')
        
        # Stream response from LLM
        def generate():
            for chunk in llm.stream(prompt):
                yield chunk
        
        return Response(generate(), mimetype='text/plain')
    
    except Exception as e:
        return Response(f"Error: {str(e)}", mimetype='text/plain')

@app.route("/admin/upload", methods=["POST"])
def admin_upload():
    """Endpoint for admins to upload new category data"""
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
        
        # Rebuild the retriever (in production, you'd want to do this asynchronously)
        global vectorstore
        try:
            vectorstore = build_retriever()
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
    app.run(debug=True, host="0.0.0.0", port=6066)