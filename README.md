# 📚 AI Chatbot with Retrieval-Augmented Generation (RAG) using Streamlit, Pinecone & OpenRouter

This project is an AI-powered chatbot that uses:
- **LangChain** for RAG (Retrieval-Augmented Generation),
- **Pinecone** as the vector database,
- **OpenRouter** as the LLM API platform (e.g., DeepSeek, Mistral, LLaMA-3),
- **Streamlit** for the web-based chat interface,
- and **PDF documents** as knowledge sources.

---

## 🧠 Features

- Upload and parse PDFs.
- Split documents into chunks for semantic search.
- Store embeddings in Pinecone.
- Retrieve relevant context and answer queries via a chat interface.
- Modular, environment-configurable LLM and embedding models (via `.env`).

---

## ⚙️ Setup Instructions

### 1. 🐍 Create & activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
