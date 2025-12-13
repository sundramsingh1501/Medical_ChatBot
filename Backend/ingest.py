import os
from dotenv import load_dotenv

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()


# CONFIG
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# 1️⃣ Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# 2️⃣ Load medical book

loader = PyPDFLoader("Data/Medical_book.pdf")
documents = loader.load()
print(f"📘 Loaded {len(documents)} pages")


# 3️⃣ Split into chunks (IMPORTANT)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=120,
    separators=["\n\n", "\n", ".", " "]
)

docs = text_splitter.split_documents(documents)
print(f"✂️ Created {len(docs)} chunks")

# 4️⃣ Embeddings (384 dim)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 5️⃣ Upload to Pinecone
PineconeVectorStore.from_documents(
    documents=docs,
    embedding=embeddings,
    index_name=INDEX_NAME
)

print("✅ Medical book successfully ingested into Pinecone")
