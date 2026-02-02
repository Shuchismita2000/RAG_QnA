from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader



def load_pdfs(data_dir: str):
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"DATA_DIR not found: {data_path}")

    pdf_paths = sorted(p for p in data_path.rglob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found in: {data_path}")

    docs = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(str(pdf_path))
        docs.extend(loader.load())
    return docs