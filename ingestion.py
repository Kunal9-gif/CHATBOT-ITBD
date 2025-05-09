import os
import time
import requests
from dotenv import load_dotenv

from pinecone import Pinecone, ServerlessSpec
from langchain_core.embeddings import Embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# --- Custom OpenRouter Embeddings class ---
class OpenRouterEmbeddings(Embeddings):
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.api_url = "https://openrouter.ai/api/v1/embeddings"

    def embed_documents(self, texts):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(
            self.api_url,
            headers=headers,
            json={"model": self.model, "input": texts},
        )
        response.raise_for_status()
        data = response.json()
        return [item["embedding"] for item in data["data"]]

    def embed_query(self, text):
        return self.embed_documents([text])[0]

# Initialize Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = os.environ.get("PINECONE_INDEX_NAME")

# Create index if not exists
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=1024,  # DeepSeek model uses 1024
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)

# Initialize embeddings
embeddings = OpenRouterEmbeddings(
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    model=os.environ.get("EMBEDDING_MODEL")
)

# Connect vector store
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Load PDF documents
loader = PyPDFDirectoryLoader("documents/")
raw_documents = loader.load()

# Split documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=400,
    length_function=len,
    is_separator_regex=False,
)
documents = text_splitter.split_documents(raw_documents)

# Generate UUIDs
uuids = [f"id{i}" for i in range(1, len(documents) + 1)]

# Add documents to vector store
vector_store.add_documents(documents=documents, ids=uuids)
print("✅ Documents successfully embedded and uploaded to Pinecone.")
