import os
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter


CONTENT_DIR = "../mentor_content"
OUTPUT_FILE = "../processed_chunks.jsonl"

def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def detect_type(filename, folder):
    if "lessons" in folder:
        return "lesson"
    if "frameworks" in folder:
        return "framework"
    if filename == "faq.csv":
        return "faq"
    if filename == "scenarios.json":
        return "scenario"
    if filename == "glossary.csv":
        return "glossary"
    return "unknown"

def process_content():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
    )

    chunks_out = []

    for root, dirs, files in os.walk(CONTENT_DIR):
        for file in files:
            path = os.path.join(root, file)

            # Skip hidden files
            if file.startswith("."):
                continue

            print(f"Processing: {path}")

            file_text = load_text(path)
            file_chunks = splitter.split_text(file_text)

            file_type = detect_type(file, root)

            for i, chunk in enumerate(file_chunks):
                metadata = {
                    "file": file,
                    "folder": os.path.basename(root),
                    "type": file_type,
                    "chunk_index": i
                }

                chunks_out.append({
                    "id": f"{file}_{i}",
                    "text": chunk,
                    "metadata": metadata
                })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in chunks_out:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n✅ Done! {len(chunks_out)} chunks created.")
    print(f"📄 Output saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    process_content()
