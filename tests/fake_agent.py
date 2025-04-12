from langchain_community.chat_models import FakeListChatModel
from langchain_core.messages import AIMessage

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


fake_responses = [
    "The weather in San Francisco is sunny with a high of 68°F.",
]

fake_llm = FakeListChatModel(responses=fake_responses)

class FakeState(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]

def fake_chatbot(state: FakeState):
    return {"messages": [fake_llm.invoke(state["messages"])]}

graph_builder = StateGraph(FakeState)
graph_builder.add_node("chatbot", fake_chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

def build_fake_agent_graph(**kwargs):
    """
    Build the fake graph for testing purposes.
    """

    return graph_builder.compile(**kwargs)

if __name__ == "__main__":
    print("Building the fake agent graph...")
    graph = build_fake_agent_graph()
    print("Graph built successfully.")
    print("Invoking the graph with a test message...")
    result = graph.invoke({"messages": [("human", "what's the weather in sf")]})
    print(result)