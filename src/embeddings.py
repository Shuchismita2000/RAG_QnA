from langchain_google_genai import GoogleGenerativeAIEmbeddings
from .config import EMBEDDING_MODEL


def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)