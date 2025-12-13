import os
import threading
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

# LOAD ENV
load_dotenv()

# CONFIGURE GEMINI (FAST MODEL)
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

# STREAMLIT CONFIG
st.set_page_config(
    page_title="Medical AI Chatbot",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Medical AI Chatbot")

st.warning(
    "⚠️ This chatbot is for educational purposes only and is NOT a substitute for professional medical advice."
)

# BACKGROUND WARM-UP (CRITICAL FIX)
def warm_up_vectorstore():
    try:
        from pinecone import Pinecone
        from langchain_pinecone import PineconeVectorStore
        from langchain_huggingface import HuggingFaceEmbeddings

        Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        st.session_state.vectorstore = PineconeVectorStore.from_existing_index(
            index_name=os.getenv("PINECONE_INDEX_NAME"),
            embedding=embeddings
        )
    except Exception:
        pass


# Start background warmup ONCE (non-blocking)
if "vectorstore" not in st.session_state:
    threading.Thread(target=warm_up_vectorstore, daemon=True).start()


def get_vectorstore():
    if "vectorstore" not in st.session_state:
        warm_up_vectorstore()
    return st.session_state.vectorstore


# RAG FUNCTION
def ask_medical_bot(question: str) -> str:
    vectorstore = get_vectorstore()

    docs = vectorstore.similarity_search(question, k=3)

    if not docs:
        return "I don't know"

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
You are a medical assistant.
Answer ONLY using the context below.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""

    response = gemini_model.generate_content(prompt)
    return response.text.strip()


# CHAT UI
if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.chat_input("Ask a medical question...")

if user_input:
    st.session_state.messages.append(("user", user_input))

    answer = ask_medical_bot(user_input)

    st.session_state.messages.append(("assistant", answer))

# DISPLAY CHAT
for role, message in st.session_state.messages:
    st.chat_message(role).write(message)
