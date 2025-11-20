import chromadb
from sentence_transformers import SentenceTransformer
from ollama import Client

# --------- CONFIG ---------
DB_DIR = "../vector_db"
TOP_K = 4
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "phi3"   # or "mistral"

# --------- LOAD COMPONENTS ---------
client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_collection("wealthplay_mentor")
embed_model = SentenceTransformer(MODEL_NAME)
ollama = Client()

SYSTEM_PROMPT = """
You are WEALTHPLAY — a friendly financial mentor for beginners.
Tone: simple, supportive, calm.
Avoid complex jargon unless explaining it.
Avoid risky trading advice (crypto, F&O, day trading).
Encourage budgeting, SIPs, emergency funds, and long-term thinking.

Response Format:
1) Acknowledge emotion calmly
2) Simple explanation
3) Small example
4) One actionable next step
5) Encouraging closing line
"""

def generate_response(user_input):
    # ---- Retrieve relevant chunks ----
    embedding = embed_model.encode(user_input).tolist()

    results = collection.query(
        query_embeddings=[embedding],
        n_results=TOP_K
    )

    retrieved_text = "\n\n".join(results["documents"][0])

    full_prompt = f"""
{SYSTEM_PROMPT}

User question: {user_input}

Relevant knowledge:
{retrieved_text}

Now answer as the mentor:
"""

    res = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": full_prompt}]
    )

    return res["message"]["content"]
