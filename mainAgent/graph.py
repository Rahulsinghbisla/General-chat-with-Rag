from langgraph.graph import StateGraph,START,END
from Rag.state import llm_cls
from Rag.graph import workflow
from General.general import workflow1
from mainAgent.node import supervisior,router


graph1 = StateGraph(llm_cls)

graph1.add_node("supervisior",supervisior)
graph1.add_node("general",workflow1)
graph1.add_node("rag",workflow)

graph1.add_edge(START,"supervisior")
graph1.add_conditional_edges("supervisior",router,
    {
        "rag": "rag",
        "general": "general",
    }
)

graph1.add_edge("rag",END)
graph1.add_edge("general",END)



chatbot = graph1.compile()

# while(True):
#     user = input("Enter your Querry : ")
#     res = chatbot.invoke({'messages':user})
#     print(f"Ai : {res['messages'][-1].content}")
#     # print(f"verdict is : {res['verdict']}")