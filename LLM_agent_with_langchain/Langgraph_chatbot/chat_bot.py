

from typing import List, Dict, Any
from langgraph.graph import StateGraph, START, END
from langchain_ollama.llms import OllamaLLM

class State(Dict):
    messages: List[Dict[str, Any]]
    
graph_builder = StateGraph(State)

llm = OllamaLLM(model="llama3.2")

def chatbot(state: State):
    # Ensure the messages are in the correct format for the LLM
    formatted_messages = [{"role": msg["role"], "content": msg["content"]} for msg in state["messages"]]
    response = llm.invoke(formatted_messages)
    # Append the assistant's response to the state
    state["messages"].append({"role": "assistant", "content": response})
    return {"messages": state["messages"]}

# Build the simple graph
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# Compile the graph
graph = graph_builder.compile()

def stream_graph_updates(user_input: str):
    # Initialize the state with the user's input
    state = {"messages": [{"role": "user", "content": user_input}]}
    # Stream updates from the graph
    for event in graph.stream(state):
        for value in event.values():
            print("Assistant: ", value["messages"][-1]["content"])

if __name__ == "__main__":
    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            break
        stream_graph_updates(user_input)