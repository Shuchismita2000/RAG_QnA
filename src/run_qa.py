import os
import sys
from pathlib import Path
from .config import DATA_DIR, INDEX_DIR
from .loaders import load_pdfs
from .chunking import chunk_documents
from .vectorstore import build_faiss_index, load_faiss_index
from .qa_chain import build_qa_chain


def ensure_index():
    if Path(INDEX_DIR).exists() and any(Path(INDEX_DIR).iterdir()):
        return load_faiss_index(INDEX_DIR)

    docs = load_pdfs(DATA_DIR)
    chunks = chunk_documents(docs)
    return build_faiss_index(chunks, INDEX_DIR)


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.run_qa \"your question\"")
        sys.exit(1)

    question = " ".join(sys.argv[1:]).strip()
    if not question:
        print("Question is required.")
        sys.exit(1)

    vectorstore = ensure_index()
    qa = build_qa_chain(vectorstore)

    result = qa(question)
    answer = result.get("result", "")
    source_docs = result.get("source_documents", [])

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