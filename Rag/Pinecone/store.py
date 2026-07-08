from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone(api_key="pcsk_4mGPBD_MSvWEvMCg5uJe1p2nbLY8VqiSRwuLbKZwdRWwP54cyvBtmKXXARYhH4JH4u1mqp")


index = pc.Index("rag")

embeddings=OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = PineconeVectorStore(embedding=embeddings, index=index)
