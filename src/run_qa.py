import os
import sys
from .config import DATA_DIR
from .loaders import load_pdfs
from .chunking import chunk_documents
from .vectorstore import (
    build_pinecone_index,
    load_pinecone_index,
    pinecone_index_is_empty,
)
from .qa_chain import build_qa_chain


def ensure_index():
    if not pinecone_index_is_empty():
        return load_pinecone_index()

    docs = load_pdfs(DATA_DIR)
    chunks = chunk_documents(docs)
    return build_pinecone_index(chunks)


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.run_qa \"your question\"")
        sys.exit(1)

    question = " ".join(sys.argv[1:]).strip()
    if not question:
        print("Question is required.")
        sys.exit(1)

    vectorstore = ensure_index()
    chain, retriever = build_qa_chain(vectorstore)
    answer = chain.invoke(question)
    source_docs = retriever.invoke(question)

    print(f"Answer: {answer}\n")
    print("Sources:")
    for doc in source_docs:
        src = doc.metadata.get("source", "")
        page = doc.metadata.get("page", None)
        if page is not None:
            print(f"- {os.path.basename(src)} (Page {page})")
        else:
            print(f"- {os.path.basename(src)}")


if __name__ == "__main__":
    main()
