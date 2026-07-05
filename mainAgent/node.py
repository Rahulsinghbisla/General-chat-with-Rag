from langchain_openai import ChatOpenAI
from Rag.state import llm_cls
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Literal,Annotated

class llm_schema(BaseModel):
    output:Literal["rag","general"]

load_dotenv()

llm = ChatOpenAI()

llm_structure = llm.with_structured_output(llm_schema)
def supervisior(state:llm_cls):
    print("In the supervisior")
    querry = state['messages']
    prompt=f"""You are a SUPERVISOR AGENT in a multi-agent system.

Your job is ONLY to classify the user query into one of two categories.

Return exactly ONE token:

rag
general

Do NOT explain. Do NOT add extra text.

------------------------------------------------

Category Definitions

rag
Return "rag" ONLY if the query requires information from uploaded documents, PDFs, files, or a private knowledge base available to the system.

Examples:
- Questions asking about content from the uploaded PDF
- "according to the document"
- "from the uploaded pdf"
- "in the provided file"
- "what does the document say about..."
- "summarize the uploaded file"
- "explain section 2 of the pdf"
- "what is written in the document about X"

Use "rag" whenever the answer must come from the uploaded document or file rather than general knowledge.

------------------------------------------------

general
Return "general" if the query is ANY of the following:

1. Greetings or small talk
   Examples:
   - hi
   - hello
   - how are you
   - good morning
   - thank you

2. General knowledge
   Examples:
   - what is AI
   - explain machine learning
   - python list comprehension

3. Coding or technical questions

4. Reasoning, brainstorming, or explanations

5. Questions NOT related to CUH documents

------------------------------------------------

Important Rule

If the query does NOT clearly require CUH documents,
ALWAYS return:

general

------------------------------------------------

User Query:
{querry}
"""
    res = llm_structure.invoke(prompt)
    print(res)
    route = res.output.strip().lower()
    return {
        "route":route
    }

def router(state: llm_cls):
    return state["route"]