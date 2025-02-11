# from dotenv import load_dotenv
# from langchain.agents import initialize_agent, Tool
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.output_parsers import StrOutputParser
# import streamlit as st
# import os

# load_dotenv()
# os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# # Check if API key is set
# if not os.getenv("GOOGLE_API_KEY"):
#     st.error("GOOGLE_API_KEY is not set. Please check your .env file.")
#     st.stop()
    
# llm = ChatGoogleGenerativeAI(
#     model = "gemini-1.5-pro",
#     temperature=0,
#     max_tokens=None,
# )

# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("System","you are a chatbot"),
#         ("Human","Questions: {question}")
#     ]
# )


# st.title("Langchain Demo with GEMINI")
# input_text = st.text_input("Enter your Questions here")

# output_parser = StrOutputParser()
# chain =  prompt | llm | output_parser

# if input_text:
#     result = chain.invoke({"question":input_text})
#     st.write( result)

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import os

# Load environment variables
load_dotenv()

# Set environment variables
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Check if API key is set
if not os.getenv("GOOGLE_API_KEY"):
    st.error("GOOGLE_API_KEY is not set. Please check your .env file.")
    st.stop()

# Initialize the Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0.5,
    max_tokens=None,
)

# Define the prompt template
prompt_1 = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful chatbot."),
        ("human", "Question: {question}")
    ]
)

prompt_2 = ChatPromptTemplate.from_messages(
    [
        ("system", "You should translate the input_language to output_language"),
        ("human", "Translate {input_text} from {input_language} to {output_language}")
    ]
)

# Set up Streamlit app
st.title("Langchain Demo with GEMINI Language Translator")
input_text = st.text_input("Enter your question here")

# Define the chain
output_parser = StrOutputParser()
chain_1= prompt_1 | llm | output_parser
chain_2 = prompt_2 | llm | output_parser

# Process input and display output
if input_text:
    with st.spinner("Generating response..."):  # Add a loading spinner
        try:
            result = chain_2.invoke(
                {"input_text": input_text,"input_language": "english", "output_language": "french"}
            
                )
            st.write(result)
        except Exception as e:
            st.error(f"An error occurred: {e}")