from langchain_openai import ChatOpenAI
from Rag.state import llm_cls
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Literal
from General.node import llm
import time

class llm_schema(BaseModel):
    output:Literal["rag","general"]

load_dotenv()

llm_structure = llm.with_structured_output(llm_schema)
def supervisior(state:llm_cls):
    start = time.time()
    print("In the supervisior")
    question = state['messages'][-1].content.strip().lower()
    
    greetings = {"hi", "hello", "hey", "thanks", "thank you", "good morning"}
    if question in greetings:
        return {"route": "general"}
    
    # otherwise LLM se classify karo
    prompt = f"""Classify the query as "rag" or "general".
"rag": query needs info from uploaded documents/PDFs/CUH knowledge base.
"general": greetings, general knowledge, coding, or anything not requiring uploaded documents.
If unsure, return "general".

Query: {question}
Return only one word: rag or general."""
    
    res = llm_structure.invoke(prompt)
    print("Result in supervisior : ",res)
    route = res.output.strip().lower()
    print(f"Supervise after {time.time()-start}")
    return {"route": route}

def router(state: llm_cls):
    return state["route"]