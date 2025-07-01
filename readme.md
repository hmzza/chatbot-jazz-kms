to check Ollama serving: http://localhost:11434/api/tags

ollama pull llama3

ollama run llama3

Run Quadrant locally:
docker run -p 6333:6333 -v $(pwd)/qdrant_data:/qdrant/storage qdrant/qdrant

for removing hash and hash txt:
rm -rf faiss_index faiss_index_hash.txt
