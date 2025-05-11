import os
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
import pinecone

load_dotenv()
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV"))

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Pinecone.from_existing_index(index_name=os.getenv("PINECONE_INDEX_NAME"), embedding=embeddings)
retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 5, "score_threshold": 0.5})
