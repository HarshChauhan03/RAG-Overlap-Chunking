# 🚀 RAG with Overlapping Chunking

## 📌 Overview

This project implements a Retrieval-Augmented Generation (RAG) system enhanced with overlapping chunking and Top-K retrieval.

The system combines:

- Sentence Transformers for semantic embeddings
- FAISS for efficient vector similarity search
- Transformer-based language model for answer generation

Unlike basic RAG systems that use simple or non-overlapping chunking, this implementation uses overlapping chunks to preserve contextual continuity and improve retrieval accuracy.

---

## 🎯 Objective

The objective of this project is to:

- Split long text into overlapping chunks
- Convert chunks into vector embeddings
- Store embeddings in a FAISS index
- Retrieve Top-K relevant chunks for a query
- Combine retrieved context
- Generate a context-aware response

This approach reflects production-style RAG system design.

---

## 🧠 Why Overlapping Chunking?

In basic chunking:

- Important information may be split between chunks
- Retrieval may miss relevant context
- Answers may become incomplete

Overlapping chunking:

- Preserves contextual flow
- Reduces information loss
- Improves retrieval quality
- Produces more complete answers

This technique is commonly used in real-world document-based AI systems.

---

## ⚙️ System Workflow

1. Long text is divided into fixed-size chunks with overlap.
2. Each chunk is converted into dense embeddings.
3. Embeddings are indexed using FAISS.
4. A user query is converted into an embedding.
5. FAISS retrieves the Top-K most similar chunks.
6. Retrieved chunks are combined.
7. The combined context and query are passed to the language model.
8. The model generates the final response.

---

## 📂 Project Structure

RAG-Overlap-Chunking/

- rag_overlap.py  
- requirements.txt  
- README.md  

---

## 🛠 Technologies Used

- Python  
- FAISS (faiss-cpu)  
- Sentence Transformers  
- HuggingFace Transformers  
- PyTorch  
- NumPy  

---

## 💻 Installation

Ensure Python 3.8+ is installed.

(Optional) Create virtual environment:

python -m venv venv  

Activate environment:

Windows:  
venv\Scripts\activate  

Mac/Linux:  
source venv/bin/activate  

Install dependencies:

pip install -r requirements.txt  

---

## ▶️ How to Run

Run:

python rag_overlap.py  

The system will:

- Create overlapping text chunks  
- Generate embeddings  
- Build FAISS index  
- Accept user queries  
- Retrieve Top-K relevant chunks  
- Generate context-based responses  

---

## 🌍 Real-World Applications

- Enterprise document search  
- Chat with large PDFs  
- Research assistant systems  
- Legal document analysis  
- AI knowledge retrieval systems  

---

## 🎓 Learning Outcomes

By completing this project, you will understand:

- Chunking strategies in RAG systems
- Overlapping chunking implementation
- Vector similarity search using FAISS
- Multi-chunk context handling
- Production-style Retrieval-Augmented Generation

---

## 👨‍💻 Author

Harsh Chauhan  
AI & Data Science Enthusiast
