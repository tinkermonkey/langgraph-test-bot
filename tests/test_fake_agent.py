from fake_agent import build_fake_agent_graph

print("Building the fake agent graph...")
graph = build_fake_agent_graph()

print("Graph built successfully.")
print("Invoking the graph with a test message...")
result = graph.invoke({"messages": [("human", "what's the weather in sf")]})
print(result)