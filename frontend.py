import streamlit as st
# from backend import graph
from embed_data import load_uploaded_pdfs, embed_docs
import os

st.set_page_config(page_title="RAG Chatbot", layout="centered")
st.title("RAG Chatbot")
st.write("Upload a document and ask your questions.")

uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)


if uploaded_files and st.button("Upload"):
    print(uploaded_files[0].name)
    with st.spinner("Uploading documents ..."):
        docs = load_uploaded_pdfs(uploaded_files=uploaded_files)
        embed_docs(docs)
    st.success("Done uploading ...")
    uploaded_files = []


# Prompt Input
st.subheader("💬 Ask your question:")
user_query = st.text_input("Enter your prompt")

# Optional model controls
with st.sidebar:
    st.header("Model Parameters")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, step=0.05)
    max_retrievals = st.slider("Max Retrievals", 1, 5, 3, step=1)

# Ask and display
if st.button("🚀 Ask"):
    if not user_query.strip():
        st.warning("Please enter a valid question.")
    else:
        with st.spinner("Generating answer..."):
            print("bye")
            # response = ask_question(
            #     collection_name=collection_name,
            #     query=user_query,
            #     temperature=temperature,
            #     max_tokens=max_tokens,
            #     model=model_choice
            # )
        st.success("Answer:")
        st.write("response")
