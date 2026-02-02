import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
DATA_DIR = os.getenv("DATA_DIR", "..\\RAG Project Dataset")
INDEX_DIR = os.getenv("INDEX_DIR", ".\\faiss_index")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
RETRIEVER_K = 4

EMBEDDING_MODEL = "models/embedding-001"
LLM_MODEL = "gemini-pro"
LLM_TEMPERATURE = 0

PROMPT_TEMPLATE = """
You are a research assistant.
Answer ONLY using the provided context.
If the answer is not present, say:
\"I could not find sufficient information in the documents.\"

Context:
{context}

Question:
{question}

Answer:
"""