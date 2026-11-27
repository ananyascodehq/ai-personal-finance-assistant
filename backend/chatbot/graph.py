"""The chatbot's LangGraph graph.

Day 3 version: one step. It takes the conversation, asks the AI from
llm.py, and adds the reply to the conversation.

    START -> chat -> END
"""

from langgraph.graph import END, START, MessagesState, StateGraph

from backend.chatbot.llm import get_llm


def build_graph():
    """Get the AI ready, then wire up the graph and return it."""
    llm = get_llm()

    def chat(state: MessagesState):
        reply = llm.invoke(state["messages"])
        return {"messages": [reply]}

    graph = StateGraph(MessagesState)
    graph.add_node("chat", chat)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)
    return graph.compile()
