
import os

import streamlit as st
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(
    page_title="AI Knowledge Assistant",
    page_icon="📚"
)

st.title("📚 AI Knowledge Assistant")
st.caption("Ask questions about your company documents.")

@st.cache_resource
def load_vector_store():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    return Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )

vector_store = load_vector_store()

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)

question = st.text_input(
    "Ask a question about your documents"
)

if st.button("Ask AI") and question:

    with st.spinner("Searching documents..."):

        # 1. Retrieve relevant chunks
        docs = vector_store.similarity_search(
            question,
            k=4
        )

        # 2. Prepare context
        context = "\n\n".join(
            doc.page_content for doc in docs
        )

        # 3. Prompt
        prompt = ChatPromptTemplate.from_template(
            """
            Answer the question using only the
            context provided below.

            If the answer is not in the context,
            say that you cannot find the answer
            in the documents.

            Context:
            {context}

            Question:
            {question}
            """
        )

        # 4. Generate answer
        response = llm.invoke(
            prompt.format(
                context=context,
                question=question
            )
        )

        st.subheader("Answer")
        st.write(response.content)

        # 5. Display sources
        st.subheader("Sources")

        for doc in docs:
            st.write(
                f"Page: {doc.metadata.get('page', 'Unknown') + 1}"
                if isinstance(doc.metadata.get("page"), int)
                else "Page information unavailable"
            )
            st.caption(doc.page_content[:500])
