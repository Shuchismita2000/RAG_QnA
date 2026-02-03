from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config import GEMINI_EMBEDDING_MODEL, OPENAI_EMBEDDING_MODEL


from os import getenv

# Support multiple embedding providers via the EMBEDDING_PROVIDER env var
# Options: "google" (default), "openai", "hf" (Hugging Face)
env_provider = getenv("EMBEDDING_PROVIDER")
if env_provider:
    provider = env_provider.lower()
else:
    # prefer openai automatically if OPENAI_API_KEY exists
    provider = "openai" if getenv("OPENAI_API_KEY") else "google"

if provider == "google":
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    embeddings = GoogleGenerativeAIEmbeddings(model=GEMINI_EMBEDDING_MODEL)

elif provider == "openai":
    # OpenAI embeddings are in the langchain_openai package
    try:
        from langchain_openai import OpenAIEmbeddings
    except ImportError as e:
        raise ImportError(
            "OpenAIEmbeddings not found. Install langchain_openai with: pip install langchain-openai"
        ) from e
    # Use 1024 dimensions to match the Pinecone index
    embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL, dimensions=1024)

elif provider in ("hf", "huggingface"):
    try:
        from langchain.embeddings.huggingface import HuggingFaceEmbeddings
    except Exception:
        from langchain.embeddings import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings()

else:
    raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {provider}")

print(f"Using embedding provider: {provider}")
