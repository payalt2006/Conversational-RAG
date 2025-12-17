import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

load_dotenv()

## Set up Streamlit UI
st.title("Conversational RAG with PDF & Chat History")
st.write("Upload a PDF and chat with the model.")

api_key = st.text_input("Enter your Groq API key:", type="password")

if api_key:
    llm = ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")

    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

    ## Process PDFs
    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            temp_pdf = './temp.pdf'
            with open(temp_pdf, "wb") as f:
                f.write(uploaded_file.getvalue())
            loader = PyPDFLoader(temp_pdf)
            documents.extend(loader.load())

        ## Split documents & create vector embeddings
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        st.write(f"Processed {len(splits)} text chunks from PDFs.")
