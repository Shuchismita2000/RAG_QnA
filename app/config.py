import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# DATA_DIR: use env var if set, otherwise default to "RAG Project Dataset" at project root
_default_data_dir = Path(__file__).parent.parent / "RAG Project Dataset"
DATA_DIR = os.getenv("DATA_DIR", str(_default_data_dir))
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-qna")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "default")
PINECONE_METRIC = os.getenv("PINECONE_METRIC", "cosine")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
RETRIEVER_K = 10

EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0

PROMPT_TEMPLATE = """
You are a research assistant answering questions based on academic papers.

Use ONLY the information from the provided context.
You may combine information from multiple context passages.

When answering:
- Be concise and technically precise.
- Avoid overgeneralization beyond what is stated in the papers.
- Cite ONLY the most relevant source passages (maximum 3).

If the answer is not supported by the context, say:
"I could not find sufficient information in the provided documents."

Context:
{context}

Question:
{question}

Answer:

"""
