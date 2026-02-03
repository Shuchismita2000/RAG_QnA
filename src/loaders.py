from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

def load_pdfs(data_dir: str):
    data_path = Path(data_dir).resolve()
    
    # If path doesn't exist, try relative to cwd or src parent
    if not data_path.exists():
        # Try as relative to project root (parent of src)
        alt_path = Path(__file__).parent.parent / data_dir
        if alt_path.exists():
            data_path = alt_path.resolve()
    
    if not data_path.exists():
        raise FileNotFoundError(f"DATA_DIR not found: {data_path} (attempted: {Path(data_dir).resolve()})")

    pdf_paths = sorted(p for p in data_path.rglob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found in: {data_path}")

    docs = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(str(pdf_path))
        docs.extend(loader.load())
    return docs