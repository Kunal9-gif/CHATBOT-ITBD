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


python3 -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

2. 📦 Install dependencies

pip install -r requirements.txt
3. 🗝️ Configure environment variables
Create a .env file in the root directory:


# OpenRouter API key
OPENROUTER_API_KEY=your_openrouter_api_key

# Embedding and Chat models (choose from OpenRouter’s supported models)
EMBEDDING_MODEL=deepseek-ai/deepseek-embedding
CHAT_MODEL=meta-llama/llama-3-70b-instruct

# Pinecone settings
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=chatbot-index
4. 📄 Prepare documents
Place your .pdf files inside a folder named documents/ in the root directory.

5. 🧠 Ingest data
Run the ingestion script to create chunks, embed them, and store them in Pinecone:


python ingestion.py
6. 🚀 Run the chatbot app

streamlit run app.py
This will launch the chatbot at http://localhost:8501.

🌐 Deployment Guide (Optional)
You can deploy this app using:

🔸 Option A: Render.com (Serverless)
Create a new Web Service on Render.

Connect your GitHub repo.

Set your build command:


pip install -r requirements.txt
Set your start command:


streamlit run app.py --server.port=10000
Add your .env variables in the Render Environment Settings.

🔸 Option B: AWS EC2 (Ubuntu)

sudo apt update && sudo apt install python3-venv -y
git clone your-repo-url
cd your-repo
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py --server.port=8501 --server.enableCORS false
#Then open port 8501 in your EC2 security group settings.

📎 File Structure
graphql
Copy code
.
├── app.py                # Streamlit chat interface
├── ingestion.py          # Load, chunk, and embed PDFs into Pinecone
├── documents/            # Your PDF files go here
├── .env                  # Environment variables
├── requirements.txt
├── .gitignore
└── README.md
