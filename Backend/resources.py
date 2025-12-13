import os
import google.generativeai as genai
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# ONE-TIME INITIALIZATION
print("🔄 Initializing resources...")

# Gemini (warm once)
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

# Pinecone
Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Embeddings (heavy)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Vectorstore
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=os.getenv("PINECONE_INDEX_NAME"),
    embedding=embeddings
)

print("✅ Resources ready")
