import streamlit as st
from rag_chat import get_rag_chain

st.set_page_config(page_title="Academic RAG Assistant", layout="wide")

st.title("📘 Academic RAG-Based Question Answering Assistant")

query = st.text_input("Ask a question here:")

rag = get_rag_chain()

if query:
    with st.spinner("Thinking..."):
        answer, sources = rag(query)

    st.subheader("📌 Answer:")
    st.write(answer)

    st.subheader("📄 Sources:")
    for i, src in enumerate(sources):
        st.markdown(f"**Source {i+1} — Page {src.metadata.get('page', 'unknown')}**")
        st.write(src.page_content[:500] + "...")
