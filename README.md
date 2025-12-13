🩺 Medical AI Chatbot (RAG-based)

🔗 Live Demo (Hugging Face):
👉 https://huggingface.co/spaces/sundram1501/medical-ai-chatbot

🔗 GitHub Repository:
👉 https://github.com/sundramsingh1501/Medical_ChatBot

📌 Overview

Medical AI Chatbot is a Retrieval-Augmented Generation (RAG) based application that answers medical questions strictly from a medical textbook using Pinecone Vector Database and Google Gemini LLM.

Unlike generic chatbots, this system does not hallucinate — it retrieves relevant context from indexed medical documents before generating responses.

⚠️ Disclaimer: This chatbot is for educational purposes only and is not a substitute for professional medical advice.

🚀 Key Features

📚 Medical Textbook Grounding (RAG)

🔎 Semantic Search using Pinecone

🤖 Google Gemini (2.5 Flash) LLM

🧠 Sentence Transformers Embeddings

💬 Interactive Streamlit Chat UI

⚡ Optimized for fast response & lazy loading

☁️ Deployed on Hugging Face Spaces

🧠 Architecture (How It Works)

Medical textbook PDF is split into chunks

Chunks are converted into vector embeddings

Embeddings are stored in Pinecone

User question → semantic search

Top relevant chunks are retrieved

Gemini LLM answers strictly using retrieved context

User Question
      ↓
Pinecone Vector Search
      ↓
Relevant Medical Context
      ↓
Gemini LLM
      ↓
Final Answer (Context-Grounded)

🛠️ Tech Stack
Layer	Technology
Frontend	Streamlit
LLM	Google Gemini 2.5 Flash
Vector DB	Pinecone
Embeddings	Sentence-Transformers (MiniLM)
Framework	LangChain
Deployment	Hugging Face Spaces
Language	Python
📂 Project Structure
Medical_Chatbot/
│
├── Backend/
│   ├── ingest.py        # PDF ingestion into Pinecone
│   ├── rag.py           # RAG pipeline logic
│
├── Frontend/
│   └── app.py           # Streamlit UI
│
├── Data/
│   └── Medical_book.pdf
│
├── requirements.txt
├── README.md
└── .gitignore

⚙️ Environment Variables

Create a .env file with:

PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=medical-chatbot
GOOGLE_API_KEY=your_gemini_api_key

▶️ Run Locally
1️⃣ Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Ingest medical book (one-time)
python Backend/ingest.py

4️⃣ Start Streamlit app
streamlit run Frontend/app.py

☁️ Deployment

This project is fully deployed on Hugging Face Spaces using a Docker-based Streamlit setup.

Auto-build from GitHub

Secure environment variables

Production-ready inference

🔗 Live App:
👉 https://huggingface.co/spaces/sundram1501/medical-ai-chatbot

🎯 Why This Project Matters

Demonstrates real-world GenAI usage

Shows RAG implementation (industry standard)

Prevents hallucinations

Uses modern LLM infrastructure

Suitable for placements, internships, and interviews

🧑‍💻 Author

Kumar Sundram
🎓 B.Tech CSE, IIIT Bhagalpur
💡 AI | ML | GenAI | RAG
🔗 GitHub: https://github.com/sundramsingh1501

⭐ Future Improvements

Multi-document ingestion

Streaming responses

Source citation per answer

User chat history persistence

Authentication
