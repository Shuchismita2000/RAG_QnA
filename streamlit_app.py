import os
import streamlit as st

from src.config import DATA_DIR
from src.loaders import load_pdfs
from src.chunking import chunk_documents
from src.vectorstore import (
    build_pinecone_index,
    load_pinecone_index,
    pinecone_index_is_empty,
)
from src.qa_chain import build_qa_chain


st.set_page_config(page_title="RAG QnA Chatbot", page_icon="💬")
st.title("RAG QnA Chatbot")


def _apply_provider_settings(llm_provider: str, embedding_provider: str):
    if llm_provider:
        os.environ["LLM_PROVIDER"] = llm_provider
    if embedding_provider:
        os.environ["EMBEDDING_PROVIDER"] = embedding_provider


@st.cache_resource(show_spinner=False)
def _get_vectorstore():
    if not pinecone_index_is_empty():
        return load_pinecone_index()
    docs = load_pdfs(DATA_DIR)
    chunks = chunk_documents(docs)
    return build_pinecone_index(chunks)


@st.cache_resource(show_spinner=False)
def _get_chain_and_retriever():
    vectorstore = _get_vectorstore()
    return build_qa_chain(vectorstore)


with st.sidebar:
    st.header("Settings")
    llm_provider = st.selectbox("LLM Provider", ["openai", "google"])
    embedding_provider = st.selectbox("Embedding Provider", ["openai", "google", "hf"])
    st.caption("Providers are applied on page load. Use rerun after changes.")
    if st.button("Rerun with Settings"):
        _apply_provider_settings(llm_provider, embedding_provider)
        st.cache_resource.clear()
        st.rerun()


if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask a question about your documents")
if question:
    _apply_provider_settings(llm_provider, embedding_provider)

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            chain, retriever = _get_chain_and_retriever()
            answer = chain.invoke(question)
            docs = retriever.invoke(question)

        st.markdown(answer)
        if docs:
            st.markdown("**Sources**")
            seen = set()
            for doc in docs:
                src = doc.metadata.get("source", "") or ""
                page = doc.metadata.get("page", None)
                key = (src, page)
                if key in seen:
                    continue
                seen.add(key)
                name = os.path.basename(src)
                if page is not None:
                    st.markdown(f"- {name} (Page {page})")
                else:
                    st.markdown(f"- {name}")

    st.session_state.messages.append({"role": "assistant", "content": answer})
