Chunk size: 800 tokens  
Overlap: 150 tokens  

Reason:

• Research papers have long coherent sections

• Prevents context loss

• Balanced retrieval granularity


EMBEDDINGS: Use Google Embeddings (as instructed).

```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)
```

VECTOR STORE (FAISS = SAFE & FAST)

```python 
from langchain.vectorstores import FAISS

vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
```



Why this scores marks:

- Semantic search 

- Multi-part queries 

- Simple justification 

 PROMPT (THIS IS IMPORTANT)

Use grounded prompting (this is a rubric item).
```python
PROMPT = """
You are a research assistant.
Answer ONLY using the provided context.
If the answer is not present, say:
"I could not find sufficient information in the documents."

Context:
{context}

Question:
{question}

Answer:
"""
```

LLM (Gemini)

```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    temperature=0
)
```

Temperature 0 = factual = anti-hallucination ✔

QA CHAIN
```python
from langchain.chains import RetrievalQA

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)
```
SOURCE ATTRIBUTION (MANDATORY)

```python
result = qa("What are the two sub-layers in each Transformer encoder layer?")

answer = result["result"]
sources = [
    {
        "document": doc.metadata.get("source"),
        "page": doc.metadata.get("page")
    }
    for doc in result["source_documents"]
]
```

Output format (SHOW THIS IN NOTEBOOK):

Answer: ...
Sources:
- attention_is_all_you_need.pdf (Page 3)
