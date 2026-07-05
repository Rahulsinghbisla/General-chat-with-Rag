from langgraph.graph import StateGraph,START,END
from Rag.state import llm_cls
from General.node import chat_node

graph = StateGraph(llm_cls)

graph.add_node("chat_node",chat_node)

graph.add_edge(START,"chat_node")
graph.add_edge("chat_node",END)

workflow1=graph.compile()