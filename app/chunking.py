from langchain_text_splitters import RecursiveCharacterTextSplitter
from .config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)


def print_chunk_counts_by_doc(docs):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    counts = {}
    for chunk in chunks:
        src = chunk.metadata.get("source", "unknown")
        counts[src] = counts.get(src, 0) + 1
    for src, count in sorted(counts.items()):
        print(f"{src}: {count} chunks")
    return counts
