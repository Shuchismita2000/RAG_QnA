# RAG_QnA

Retrieval-Augmented Generation (RAG) QnA over your PDFs using LangChain + Pinecone,
with optional OpenAI or Google (Gemini) providers. Includes a CLI runner and a
Streamlit chatbot UI.

## Features
- Load PDFs from a dataset folder
- Chunk and embed documents
- Store and search vectors in Pinecone
- Chat-style QnA with sources
- Switch providers via env vars

## Demo
[sample_streamlit.mp4](sample_streamlit.mp4)

## Project Structure
- `app/` - Streamlit app + app-specific pipeline
- `notebooks/` - Demo notebook
- `RAG Project Dataset/` - PDF source data
- `requirements.txt` - Python dependencies

## Setup
1) Create and activate a virtual environment
2) Install dependencies
```bash
pip install -r requirements.txt
```

## Environment Variables
Create a `.env` file in the project root:
```
GOOGLE_API_KEY=...
OPENAI_API_KEY=...
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=rag-qna
PINECONE_NAMESPACE=default
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
DATA_DIR=..\RAG Project Dataset

# Provider selection
LLM_PROVIDER=openai
EMBEDDING_PROVIDER=openai
```

## Run (CLI)
```bash
python -m app.run_qa "Your question here"
```

## Run (Streamlit)
```bash
streamlit run app/streamlit_app.py
```

## Notes
- If you change providers or add new PDFs, consider rebuilding the index.
- `RETRIEVER_K`, `CHUNK_SIZE`, and `CHUNK_OVERLAP` live in `app/config.py`.
