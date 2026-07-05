from langgraph.graph.message import BaseMessage,add_messages
from typing import TypedDict,Annotated,List
from langchain_core.documents import Document

class llm_cls(TypedDict):
    messages:Annotated[List[BaseMessage],add_messages]
    docs:List[Document]

    good_docs:List[Document]
    verdict:str
    reason:str
    
    strips:List[str]
    keep_strip:List[str]
    refined:str

    answer:str
    route:str