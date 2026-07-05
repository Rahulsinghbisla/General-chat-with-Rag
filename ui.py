import streamlit as st
import requests
from Pinecone import load_docs

st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📚",
    layout="wide"
)

st.title("📚 PDF RAG Chatbot")
st.markdown("Upload a PDF and ask questions about it.")

API_URL = "https://general-chat-with-rag.onrender.com/chat/"

# session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "is_loading" not in st.session_state:
    st.session_state.is_loading = False

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# Sidebar
st.sidebar.header("Upload PDF")

uploaded_file = st.sidebar.file_uploader(
    "Upload your PDF",
    type="pdf"
)

if uploaded_file is not None:

    import tempfile

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    load_docs(file_path)

    st.sidebar.success("PDF Processed Successfully!")

# -------------------------------
# Display old chat history first
# -------------------------------

for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(message)

# -------------------------------
# Chat Input (disabled while loading)
# -------------------------------

query = st.chat_input(
    "Ask a question about the PDF",
    disabled=st.session_state.is_loading
)

# Step 1: user just typed something -> show it, set pending, rerun
if query and not st.session_state.is_loading:
    st.session_state.chat_history.append(("user", query))
    st.session_state.pending_query = query
    st.session_state.is_loading = True
    st.rerun()

# Step 2: if there's a pending query, show user msg + spinner, then call API
# Step 2: if there's a pending query, show spinner, then call API
if st.session_state.is_loading and st.session_state.pending_query:

    with st.chat_message("assistant"):
        with st.spinner("Thinking... 🤔"):
            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": st.session_state.pending_query
                    }
                ]
            }

            try:
                api_response = requests.post(API_URL, json=payload, timeout=60)
                api_response.raise_for_status()
                data = api_response.json()
                print(f"Response from API: {data}")

                response = data.get("response", data)
                if isinstance(response, list) and len(response) > 0:
                    response = response[-1].get("content", str(response))

            except requests.exceptions.RequestException as e:
                response = f"⚠️ Error calling chatbot API: {e}"

        st.write(response)

    st.session_state.chat_history.append(("assistant", response))
    st.session_state.pending_query = None
    st.session_state.is_loading = False
    st.rerun()
