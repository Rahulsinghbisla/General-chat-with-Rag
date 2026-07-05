from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph,START,END
from Rag.nodes import refine,retrieve,generate,llm_cls,eval_doc

graph = StateGraph(llm_cls)

graph.add_node("retrieve",retrieve)
graph.add_node("refine",refine)
graph.add_node("generate",generate)
graph.add_node("eval_doc",eval_doc)

graph.add_edge(START,"retrieve")
graph.add_edge("retrieve","eval_doc")
graph.add_edge("eval_doc","refine")
graph.add_edge("refine","generate")
graph.add_edge("refine",END)

workflow=graph.compile()
# while(True):
#     user = input("Enter your Querry : ")
#     res = workflow.invoke({'messages':user})
#     print(f"Ai : {res['answer']}")
#     print(f"verdict is : {res['verdict']}")