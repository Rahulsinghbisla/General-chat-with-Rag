from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from Rag.Pinecone.store import vector_store

def load_docs(file):
    loader = PyPDFLoader(file)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=130)

    docs = loader.load()

    text = text_splitter.split_documents(docs)
    vector_store.add_documents(text)