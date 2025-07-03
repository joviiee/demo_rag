import streamlit as st
from backend import build_agent
from embed_data import load_uploaded_pdfs, embed_docs, clear_all_pgvector_data
from langchain_core.messages import HumanMessage,AIMessage

st.set_page_config(page_title="RAG Chatbot", layout="centered")
st.title("RAG Chatbot")
st.write("Upload a document and ask your questions.")

uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)


if uploaded_files and st.button("Upload"):
    with st.spinner("Uploading documents ..."):
        docs = load_uploaded_pdfs(uploaded_files=uploaded_files)
        embed_docs(docs)
    st.success("Done uploading ...")
    uploaded_files = []


# Prompt Input
st.subheader("Ask your question:")
user_query = st.text_input("Enter your prompt")

# Optional model controls
with st.sidebar:
    st.header("Model Parameters")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, step=0.05)
    max_retrievals = st.slider("Max Retrievals", 1, 5, 3, step=1)

if "agent" not in st.session_state:
    st.session_state.agent = build_agent()

# Ask and display
if st.button("Ask"):
    if not user_query.strip():
        st.warning("Please enter a valid question.")
    else:
        with st.spinner("Generating answer..."):
            current_state = {
                "messages":[HumanMessage(content=user_query)],
                "temperature":temperature,
                "num_results":max_retrievals
            }
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                full_response = ""

                response_placeholder.empty()
                
                for step in st.session_state.agent.stream(
                    current_state,
                    config={
                        "configurable": {
                            "thread_id": "user123",
                            "temperature":temperature,
                            "num_results":max_retrievals
                        }
                    },
                    stream_mode="values"
                ):
                    msg = step["messages"][-1]
                    if isinstance(msg,AIMessage):
                        full_response += msg.content
                        response_placeholder.markdown(full_response)

if st.button("Clear DataBase"):
    clear_all_pgvector_data()