from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import sqlite3

# Template will ensure bot answer in the format below
# can have more templates for different scenarios

template = """
Answer the question Below

Here is the conversation history: {context}
Question: {question}

Answer:
"""
# use opensource model llama3.2
model = OllamaLLM(model = "llama3.2")

prompt = ChatPromptTemplate.from_template(template)  
# formation of the chain
# prompt will be used to format the conversation
chain = prompt | model


# def handle_conversation():
#     """
#     This function will handle the conversation between the user and the bot
#     Also  store those conversation in the context
#     """
#     context = ""
#     print("Welcome to the conversation!  Type 'exit' to quit")  
#     while True:
#         question = input("You: ")
#         if question.lower() == "exit":
#             break
        
#         result = chain.invoke({"context":context, "question":question})
#         context += f"\nUser: {question}\nBot: {result}"
#         print("Bot: ", result)

# # result  = chain.invoke({"context":"", "question" : "What is your name?"})   
# # print(result)
# import sqlite3

# if __name__ == "__main__":
#     handle_conversation()
    

# Initialize the database
conn = sqlite3.connect("chat_history.db")
cursor = conn.cursor()

# Create table if not exists
cursor.execute("""
CREATE TABLE IF NOT EXISTS chat (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_input TEXT,
    bot_response TEXT
)
""")
conn.commit()

def save_to_db(user_input, bot_response):
    cursor.execute("INSERT INTO chat (user_input, bot_response) VALUES (?, ?)", (user_input, bot_response))
    conn.commit()

def load_context():
    cursor.execute("SELECT user_input, bot_response FROM chat")
    history = cursor.fetchall()
    return "\n".join([f"User: {u}\nBot: {b}" for u, b in history])

def handle_conversation():
    context = load_context()  # Load previous history
    print("Welcome to the conversation! Type 'exit' to quit")

    while True:
        question = input("You: ")
        if question.lower() == "exit":
            break
        
        result = chain.invoke({"context": context, "question": question})
        context += f"\nUser: {question}\nBot: {result}"
        save_to_db(question, result)  # Save to database
        print("Bot: ", result)

if __name__ == "__main__":
    handle_conversation()
