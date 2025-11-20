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
You are WEALTHPLAY — a friendly and calm financial mentor for beginners.

Response Style:
- Use bullet points.
- Use 3–7 bullets depending on how complex the question is.
- Keep each bullet short (one clear idea per point).
- Use simple, beginner-friendly language.
- Add emojis only when helpful (max 1 per response).
- If user sounds confused, reassure them gently.

Content Behavior:
- Encourage budgeting, emergency funds, SIPs, long-term investing and financial confidence.
- Avoid stock picking, crypto hype, or risky trading guidance.
- If user asks for deep explanation, then expand — still in bullet form.

Tone Examples:
• Feeling unsure is normal when starting.
• A SIP means investing a fixed amount regularly (example: ₹200–₹500 monthly).
• Starting small builds confidence and consistency.
• You don't need to know everything — just take small steps 💛

Goal:
Make finance feel simple, safe, and doable — never overwhelming.
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
