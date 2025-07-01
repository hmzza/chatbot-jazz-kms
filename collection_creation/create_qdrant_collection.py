from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import uuid

# Connect to Qdrant
client = QdrantClient(host="localhost", port=6333)

# Use BAAI/bge-small-en-v1.5 with sentence-transformers
model = SentenceTransformer("BAAI/bge-small-en-v1.5")
# Documents to embed
documents = [
    "Jazz is Pakistan's leading digital operator providing internet, call, and SMS services.",
    "JazzCash is Jazz's mobile wallet service offering bill payment, mobile load, and money transfers.",
    "Jazz offers corporate solutions like IoT, cloud services, and managed security for businesses.",
    "Jazz provides affordable prepaid and postpaid packages to millions of users across Pakistan.",
    "Jazz bundles include voice minutes, SMS, internet data, and international calling options.",
    "Jazz is a trusted provider in Pakistan for both individuals and businesses seeking telecom solutions."
]

# Generate embeddings
vectors = model.encode(documents).tolist()

# Create payloads (metadata)
payloads = [{"text": doc} for doc in documents]

# Upload to Qdrant
client.recreate_collection(
    collection_name="jazz_rag_vectorstore",
    vectors_config=models.VectorParams(size=len(vectors[0]), distance=models.Distance.COSINE)
)

client.upsert(
    collection_name="jazz_rag_vectorstore",
    points=[
        models.PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload=payload
        )
        for vector, payload in zip(vectors, payloads)
    ]
)

print("✅ Qdrant collection created and data uploaded successfully!")
