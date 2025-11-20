import chromadb
from sentence_transformers import SentenceTransformer

DB_DIR = "../vector_db"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Load model
model = SentenceTransformer(MODEL_NAME)

# Connect to DB
client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_collection("wealthplay_mentor")

def get_response(query, k=5):
    embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[embedding],
        n_results=k
    )

    print("\n🔍 Query:", query)
    print("-" * 50)

    for i in range(len(results["ids"][0])):
        print(f"\n📌 Result {i+1}:")
        print("ID:", results["ids"][0][i])
        print("Text:", results["documents"][0][i][:300], "...")
        print("Metadata:", results["metadatas"][0][i])
        print("-" * 50)

while True:
    q = input("\nAsk something (or type EXIT): ")
    if q.lower() == "exit":
        break
    get_response(q)
