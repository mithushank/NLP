from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

template = """
Answer the question Below

Here is the conversation history: {context}
Question: {question}

Answer:
"""

model = OllamaLLM(model = "llama3.2")
prompt = ChatPromptTemplate.from_template(template)  
chain = prompt | model

def handle_conversation():
    context = ""
    print("Welcome to the conversation!  Type 'exit' to quit")  
    while True:
        question = input("You: ")
        if question.lower() == "exit":
            break
        
        result = chain.invoke({"context":context, "question":question})
        context += f"\nUser: {question}\nBot: {result}"
        print("Bot: ", result)

# result  = chain.invoke({"context":"", "question" : "What is your name?"})   
# print(result)

if __name__ == "__main__":
    handle_conversation()