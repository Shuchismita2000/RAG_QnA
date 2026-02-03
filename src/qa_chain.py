from os import getenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from .config import PROMPT_TEMPLATE, LLM_MODEL, LLM_TEMPERATURE, RETRIEVER_K


def _get_llm():
    # Support multiple LLM providers via the LLM_PROVIDER env var
    # Options: "google" (default), "openai"
    env_provider = getenv("LLM_PROVIDER")
    if env_provider:
        provider = env_provider.lower()
    else:
        # prefer openai automatically if OPENAI_API_KEY exists
        provider = "openai" if getenv("OPENAI_API_KEY") else "google"

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
    elif provider == "openai":
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")

    print(f"Using LLM provider: {provider}")
    return llm


def build_qa_chain(vectorstore):
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )
    llm = _get_llm()
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever
