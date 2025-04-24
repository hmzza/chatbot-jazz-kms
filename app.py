
from flask import Flask, request, render_template, Response
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

app = Flask(__name__)

# 1. Load documents from cleaned_data folder
def load_documents():
    docs = []
    for file in os.listdir("cleaned_data"):
        if file.endswith(".txt"):
            loader = TextLoader(os.path.join("cleaned_data", file))
            docs.extend(loader.load())
    return docs

# 2. Build retriever with FAISS and BGE embeddings
def build_retriever():
    docs = load_documents()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    return retriever

retriever = build_retriever()
llm = Ollama(model="llama3", temperature=0.7)

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

        # Try retrieving documents
        docs = retriever.get_relevant_documents(user_input)
        context = "\n".join([doc.page_content for doc in docs if doc.page_content.strip()])

        # If relevant context exists, use RAG; otherwise, fallback to plain chatbot
        if context and len(context.split()) > 30:
            prompt = f"""You are JazzBot, a helpful assistant for Jazz Telecom (Pakistan).
You are answering based on the following internal knowledge:
{context}

Answer this user query in detail:
{user_input}
"""
        else:
            prompt = user_input

        # Stream response from LLM
        def generate():
            for chunk in llm.stream(prompt):
                yield chunk

        return Response(generate(), mimetype='text/plain')

    except Exception as e:
        return Response(f"Error: {str(e)}", mimetype='text/plain')

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=6065)
