from pathlib import Path
from langchain.vectorstores import FAISS
from .embeddings import get_embeddings


def build_faiss_index(docs, index_dir: str):
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    Path(index_dir).mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(index_dir)
    return vectorstore


def load_faiss_index(index_dir: str):
    embeddings = get_embeddings()
    return FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)