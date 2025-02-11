from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI  # Use OpenAI LLM
from langchain_community.utilities import GoogleSerperAPIWrapper  # Use Serper wrapper
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
api_key=os.getenv("OPENAI_API_KEY")
# Initialize Serper API wrapper
search_tool = GoogleSerperAPIWrapper(serper_api_key=os.getenv("SERPER_API_KEY"))

# Initialize OpenAI LLM
llm = OpenAI( temperature=0.5)

# Define tools
tools = [
    Tool(
        name="search",
        func=search_tool.run,  # Use the `run` method of the Serper wrapper
        description="Search the web for information"
    )
]

# Initialize the agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",  # Correct agent type
    verbose=True
)

# Run the agent
response = agent.run("What is the capital of France?")
print(response)
