from fastapi import FastAPI
from pydantic import BaseModel
from rag import ask_medical_bot  # ✅ FIXED IMPORT

app = FastAPI(
    title="Medical Chatbot API",
    description="RAG-based Medical Assistant using Gemini + Pinecone"
)

# Request Schema
class Query(BaseModel):
    question: str

# API Endpoint
@app.post("/ask")
async def ask(query: Query):
    answer = ask_medical_bot(query.question)
    return {"answer": answer}
