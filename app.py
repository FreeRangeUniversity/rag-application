import streamlit as st
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings


from config import COLLECTION_NAME, QDRANT_PATH

st.title("📄 Local PDF RAG")

# Set up
client = QdrantClient(path=QDRANT_PATH)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vectordb = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME, embedding=embeddings)
retriever = vectordb.as_retriever(search_kwargs={"k": 5})
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

query = st.text_input("Ask a question:")
if query:
    docs = retriever.invoke(query)

    st.write(f"Retrieved {len(docs)} chunks")

    # Display chunks that were a part of retrieval
    # for i, d in enumerate(docs[:5]):
    #   st.write(f"Chunk {i + 1}:")
    #   st.write(d.page_content[:500])


    context = "\n\n".join(d.page_content for d in docs)

    prompt = f"""
You are a helpful assistant.

Use the context below to answer the question.
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{query}
"""

    response = llm.invoke(prompt)

    st.write(response.content)
