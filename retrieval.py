import os
import requests
from dotenv import load_dotenv

from pinecone import Pinecone
from langchain_core.embeddings import Embeddings
from langchain_pinecone import PineconeVectorStore

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
index = pc.Index(index_name)

# Initialize embeddings using OpenRouter
embeddings = OpenRouterEmbeddings(
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    model=os.environ.get("EMBEDDING_MODEL")
)

# Connect to vector store
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Create retriever
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 5, "score_threshold": 0.5},
)

# Run retrieval query
query = "what is retrieval augmented generation?"
results = retriever.invoke(query)

# Show results
print("RESULTS:\n")
for res in results:
    print(f"* {res.page_content} [{res.metadata}]")
