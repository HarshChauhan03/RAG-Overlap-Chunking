# RAG with Overlapping Chunking
# Author: Harsh Chauhan

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# -----------------------------------
# 1️⃣ Load Models
# -----------------------------------

print("Loading models...")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
generator = pipeline("text-generation", model="gpt2")

print("Models loaded successfully.\n")

# -----------------------------------
# 2️⃣ Sample Knowledge Base (Long Text)
# -----------------------------------

text = """
Artificial Intelligence is transforming industries worldwide.
Machine learning is a subset of Artificial Intelligence.
It enables systems to learn from data without explicit programming.
Deep learning uses neural networks with multiple hidden layers.
Natural Language Processing allows machines to understand text and speech.
Supervised learning requires labeled data.
Unsupervised learning discovers hidden patterns in data.
Reinforcement learning learns by interacting with the environment.
"""

# -----------------------------------
# 3️⃣ Overlapping Chunking Function
# -----------------------------------

def create_chunks(text, chunk_size=200, overlap=50):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += chunk_size - overlap

    return chunks


documents = create_chunks(text)

print(f"Total Chunks Created: {len(documents)}\n")

# -----------------------------------
# 4️⃣ Create Embeddings
# -----------------------------------

doc_embeddings = embedding_model.encode(documents)
doc_embeddings = np.array(doc_embeddings).astype("float32")

# -----------------------------------
# 5️⃣ Create FAISS Index
# -----------------------------------

dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)

print("🚀 RAG with Overlapping Chunking Ready!")
print("Type 'quit' to exit.\n")

# -----------------------------------
# 6️⃣ RAG Loop
# -----------------------------------

while True:
    query = input("Ask a question: ")

    if query.lower() == "quit":
        print("Exiting...")
        break

    # Encode query
    query_vector = embedding_model.encode([query])
    query_vector = np.array(query_vector).astype("float32")

    # Retrieve Top-K chunks
    k = 3
    distances, indices = index.search(query_vector, k)

    retrieved_chunks = [documents[i] for i in indices[0]]

    combined_context = "\n".join(retrieved_chunks)

    # Build prompt
    prompt = f"""
You are an AI assistant.

Context:
{combined_context}

Question:
{query}

Answer clearly based only on the context above.
Answer:
"""

    # Generate response
    output = generator(prompt, max_length=200, temperature=0.7)
    answer = output[0]["generated_text"]

    print("\n📌 Retrieved Chunks:")
    for chunk in retrieved_chunks:
        print("-", chunk)

    print("\n🧠 Generated Answer:")
    print(answer)
    print("-" * 60)