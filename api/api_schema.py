from pydantic import BaseModel
from typing import List, Literal

class TestResponse(BaseModel):
    status: str = "ok"

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]   

class ChatReply(BaseModel):
    response: List[ChatMessage]