from flask import Flask, request, render_template, Response
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
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
    
    # For now, we'll use the directory structure to infer categories
    # Later you can set up a proper directory structure like:
    # cleaned_data/products/offers.txt, cleaned_data/vas/offers.txt, etc.
    
    # Current implementation - add category metadata to existing files
    for file in os.listdir("cleaned_data"):
        if file.endswith(".txt"):
            # Try to determine category from filename
            category = "products"  # Default category
            file_type = None
            
            if "offers" in file:
                file_type = "offers"
            elif "packages" in file:
                file_type = "packages"
            elif "data_offers" in file:
                file_type = "data_offers"
            
            loader = TextLoader(os.path.join("cleaned_data", file))
            documents = loader.load()
            
            # Add metadata to each document
            for doc in documents:
                doc.metadata["category"] = category
                doc.metadata["file_type"] = file_type
                doc.metadata["filename"] = file
            
            docs.extend(documents)
    
    return docs

# 2. Build retriever with FAISS, BGE embeddings, and metadata filtering
def build_retriever():
    docs = load_documents_with_categories()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    
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
    
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Return the vectorstore directly instead of as_retriever for more flexibility
    return vectorstore

# Initialize resources
vectorstore = build_retriever()
llm = Ollama(model="llama3", temperature=0.7)

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
        
        # Detect if the query is targeting a specific category
        detected_category = detect_category(user_input)
        
        # Retrieve relevant documents
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
        vectorstore = build_retriever()
        
        return Response("File uploaded successfully", status=200)
    
    except Exception as e:
        return Response(f"Error: {str(e)}", status=500)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=6066)