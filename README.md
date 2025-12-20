# Conversational-RAG-with-PDF-Chat-History
This project is a Conversational RAG (Retrieval-Augmented Generation) system that allows users to upload PDFs, chat with an AI assistant, and retrieve context-aware responses using Groq's LLM, FAISS, ChromaDB, and Hugging Face embeddings.
# 📖 Conversational RAG with PDF & Chat History 🚀  

This project is a **Conversational RAG (Retrieval-Augmented Generation) system** that allows users to **upload PDFs, chat with an AI assistant**, and **retrieve context-aware responses** using **Groq's LLM, FAISS, ChromaDB, and Hugging Face embeddings**.  

---

## ✨ Features  
✅ **Upload & Process PDFs** – Extract text from PDFs and create embeddings.  
✅ **Context-Aware Chat** – Uses session-based memory for conversational understanding.  
✅ **Groq API Integration** – Generates responses using the **Gemma2-9b-It** model.  
✅ **Vector Search with ChromaDB & FAISS** – Retrieves relevant document chunks for better responses.  
✅ **Session-Based Chat History** – Maintains chat history across user queries.  
✅ **Fast & Lightweight** – Built with **Streamlit** for a simple UI.  

---

## 🛠️ Installation & Setup  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/your-username/conversational-rag.git
cd conversational-rag
```
### 2️⃣ Create a Virtual Environment (Optional but Recommended)
```bash

Edit
python -m venv venv
source venv/bin/activate   # On macOS/Linux
venv\Scripts\activate      # On Windows
```
### 3️⃣ Install Dependencies
```bash

pip install -r requirements.txt
```
### 4️⃣ Set Up Environment Variables
Create a .env file in the root directory and add the following:

```ini

HF_TOKEN="your-huggingface-token"
GROQ_API_KEY="your-groq-api-key"
```
Replace your-huggingface-token and your-groq-api-key with your actual API keys.

### 5️⃣ Run the Application
```bash

streamlit run app.py
```
Then, open the Streamlit app in your browser to start chatting with your PDF-based AI assistant!

### 📁 Project Structure
```bash

📂 conversational-rag
│-- 📜 app.py                  # Main Streamlit application
│-- 📜 requirements.txt        # Required dependencies
│-- 📜 .env                    # Environment variables (ignored in .gitignore)
│-- 📜 README.md               # Project documentation
│-- 📂 chroma_db               # Stores vector embeddings (created at runtime)
│-- 📂 temp.pdf                # Temporary storage for uploaded PDFs
```
### 🎯 How to Use
1️⃣ Upload a PDF using the UI.
2️⃣ Enter your Groq API key to authenticate the model.
3️⃣ Ask questions based on the document’s content.
4️⃣ The AI retrieves relevant information and generates answers.
5️⃣ Chat history is stored for a natural conversation flow.

### 📦 Dependencies
This project requires the following Python packages:

streamlit
langchain
langchain_groq
langchain_openai
langchain_huggingface
langchain_chroma
langchain_community
chromadb
faiss-cpu
pypdf
python-dotenv
To install all dependencies, run:

```bash

pip install -r requirements.txt
```
