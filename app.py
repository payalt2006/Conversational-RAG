import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

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

        vectorstore = Chroma.from_documents(splits, embedding=embeddings, persist_directory="./chroma_db")
        retriever = vectorstore.as_retriever()

        ## Setup Answer Generation
        system_prompt = (
            "You are an assistant for answering questions. "
            "Use the retrieved context to provide accurate responses. "
            "If you don’t know the answer, say so. "
            "Keep answers concise (max 3 sentences).\n\n{context}"
        )

        qa_context = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])

        question_answer_chain = create_stuff_documents_chain(llm, qa_context)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        st.write("RAG chain setup completed successfully.")
