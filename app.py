import os
import streamlit as st
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from retriever import retriever

load_dotenv()

st.title("💡 Company Knowledge ChatBot")
prompt = st.chat_input("Ask me anything about our company policies...")

if prompt:
    llm = ChatOpenAI(model="openai/gpt-4o", api_key=os.getenv("OPENROUTER_API_KEY"))
    template = PromptTemplate.from_template("""
    You are a helpful assistant. Use the following context to answer the question:
    {context}
    Question: {question}
    Answer:
    """)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=False)
    response = qa_chain.run(prompt)
    st.markdown(response)
