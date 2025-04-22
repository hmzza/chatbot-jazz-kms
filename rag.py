from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

def get_qa_chain():
    loader = TextLoader("cleaned_data/cleaned_packages.txt")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    client = QdrantClient(host="localhost", port=6333)

    vectorstore = Qdrant.from_documents(
        chunks,
        embedding=embeddings,
        location="http://localhost:6333",
        collection_name="jazz_data",
        force_recreate=True,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = Ollama(model="llama3", temperature=0.2)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
    )

    return qa_chain
