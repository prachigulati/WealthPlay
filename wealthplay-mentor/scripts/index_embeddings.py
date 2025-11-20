import os
import json
from sentence_transformers import SentenceTransformer
import chromadb

# --- CONFIG ---
CHUNKS_FILE = "../processed_chunks.jsonl"
DB_DIR = "../vector_db"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- LOAD EMBEDDING MODEL ---
print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)

# --- NEW CHROMA CLIENT SYNTAX ---
client = chromadb.PersistentClient(path=DB_DIR)

# Create or load collection
collection = client.get_or_create_collection(
    name="wealthplay_mentor",
    metadata={"hnsw:space": "cosine"}  # similarity metric
)

# --- READ CHUNKS ---
print("Loading chunks...")

records = []
with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

print(f"Loaded {len(records)} records.")

# --- PROCESS & ADD ---
print("Generating embeddings & indexing...")

for rec in records:
    text = rec["text"]
    metadata = rec["metadata"]
    chunk_id = rec["id"]

    embedding = model.encode(text).tolist()

    # prevent duplicates
    try:
        collection.add(
            documents=[text],
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[chunk_id],
        )
    except Exception:
        print(f"⚠️ Skipped duplicate: {chunk_id}")

print("\n🎉 Embeddings stored successfully!")
print(f"📁 Vector DB saved in: {DB_DIR}")
