from api.api_schema import TestResponse,ChatRequest,ChatReply,ChatMessage
from api.utlies import to_langchain_messages
from fastapi import FastAPI, HTTPException
from mainAgent.graph import chatbot


app = FastAPI(title="Corrective RAG", version="1.0")

@app.get("/test", response_model=TestResponse, summary="testing", description="Check the testing of the API.")
def health_check():
    return TestResponse(status="ok")

@app.post("/chat", summary="Query the RAG system", description="Send a query to the RAG system and receive a response.")
def query_rag(request: ChatRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty")
    changed_messages = to_langchain_messages(request.messages)
    try:
        res = chatbot.invoke({"messages": changed_messages})
        response = res["messages"][-1].content
        return ChatReply(response=[ChatMessage(role="assistant", content=response)])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    