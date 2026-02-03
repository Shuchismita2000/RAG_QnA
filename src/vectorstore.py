from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from .embeddings import get_embeddings
from .config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE,
    PINECONE_METRIC,
    PINECONE_CLOUD,
    PINECONE_REGION,
)


def _get_pinecone_client():
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY is not set.")
    if not PINECONE_INDEX_NAME:
        raise ValueError("PINECONE_INDEX_NAME is not set.")
    return Pinecone(api_key=PINECONE_API_KEY)


def _get_or_create_index(dimension: int):
    pc = _get_pinecone_client()
    existing = pc.list_indexes().names()
    if PINECONE_INDEX_NAME not in existing:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=dimension,
            metric=PINECONE_METRIC,
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
    return pc.Index(PINECONE_INDEX_NAME)


def pinecone_index_is_empty():
    pc = _get_pinecone_client()
    index = pc.Index(PINECONE_INDEX_NAME)
    stats = index.describe_index_stats()
    return stats.get("total_vector_count", 0) == 0


def build_pinecone_index(docs):
    embeddings = get_embeddings()
    dimension = len(embeddings.embed_query("dimension_probe"))
    _get_or_create_index(dimension)
    return PineconeVectorStore.from_documents(
        docs,
        embeddings,
        index_name=PINECONE_INDEX_NAME,
        namespace=PINECONE_NAMESPACE,
    )


def load_pinecone_index():
    embeddings = get_embeddings()
    return PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
        namespace=PINECONE_NAMESPACE,
    )
