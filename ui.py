import streamlit as st
from Rag.Pinecone.Pinecone import load_docs
from mainAgent.graph import chatbot

st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📚",
    layout="wide"
)

st.title("📚 PDF RAG Chatbot")
st.markdown("Upload a PDF and ask questions about it.")

# session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar
st.sidebar.header("Upload PDF")

uploaded_file = st.sidebar.file_uploader(
    "Upload your PDF",
    type="pdf"
)

if uploaded_file is not None:

    import tempfile

    # create temporary file
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
# Chat Input
# -------------------------------

query = st.chat_input("Ask a question about the PDF")

if query:

    # show user message immediately
    with st.chat_message("user"):
        st.write(query)

    st.session_state.chat_history.append(("user", query))

    # generate response
    res = chatbot.invoke({"messages": query})
    response = res["messages"][-1].content
    print(f"Response in querry of frontent : {response}")

    # show assistant response
    with st.chat_message("assistant"):
        st.write(response)

    st.session_state.chat_history.append(("assistant", response))