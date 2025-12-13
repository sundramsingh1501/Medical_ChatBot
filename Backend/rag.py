import os
from dotenv import load_dotenv
import google.generativeai as genai

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

# LOAD ENV
load_dotenv()

# CONFIG
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ONE-TIME INITIALIZATION (FAST)

# Gemini 2.5 (official SDK)
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

# Pinecone
Pinecone(api_key=PINECONE_API_KEY)

# Embeddings (heavy → load once)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Vectorstore (existing index)
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings
)

# RAG FUNCTION (FAST PATH)
def ask_medical_bot(question: str) -> str:
    docs = vectorstore.similarity_search(question, k=4)

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
