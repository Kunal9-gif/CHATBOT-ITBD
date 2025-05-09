import os
import streamlit as st
import requests
from dotenv import load_dotenv

# Pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# LangChain messages
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

# Load environment variables
load_dotenv()

st.title("Chatbot with OpenRouter + Pinecone")

# ---------------------
# Custom OpenRouter Embeddings
# ---------------------
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

# ---------------------
# Custom OpenRouter Chat LLM
# ---------------------
class OpenRouterChat(BaseChatModel):
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"

    def _convert_messages(self, messages):
        return [{"role": msg.type, "content": msg.content} for msg in messages]

    def invoke(self, messages, **kwargs):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": self._convert_messages(messages),
        }
        response = requests.post(self.api_url, headers=headers, json=data)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return AIMessage(content)

# ---------------------
# Pinecone setup
# ---------------------
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = os.environ.get("PINECONE_INDEX_NAME")
index = pc.Index(index_name)

# ---------------------
# Embeddings + Vector Store
# ---------------------
embeddings = OpenRouterEmbeddings(
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    model=os.environ.get("EMBEDDING_MODEL")
)

vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# ---------------------
# Chat history
# ---------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append(SystemMessage("You are an assistant for question-answering tasks."))

# Display previous messages
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# ---------------------
# Prompt input
# ---------------------
prompt = st.chat_input("Ask a question...")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append(HumanMessage(prompt))

    # LLM instance
    llm = OpenRouterChat(
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        model=os.environ.get("CHAT_MODEL")
    )

    # Retriever
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.5},
    )

    docs = retriever.invoke(prompt)
    docs_text = "".join(d.page_content for d in docs)

    # System prompt with context
    system_prompt = f"""You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.
Context: {docs_text}"""

    st.session_state.messages.append(SystemMessage(system_prompt))

    # Invoke OpenRouter
    result = llm.invoke(st.session_state.messages).content

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(result)
    st.session_state.messages.append(AIMessage(result))
