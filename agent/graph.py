import os
import dotenv
import logging
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph_sdk import Auth
from agent.checkpointer import get_postgres_checkpointer
import asyncio

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dotenv.load_dotenv(".env", override=True)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "langgraph-test-bot"

tools = [TavilySearchResults(max_results=1)]

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = (
    "You are a helpful assistant. "
    "You may not need to use tools for every query - the user may just want to chat!"
)

# Get the PostgreSQL checkpointer
checkpointer = asyncio.run(get_postgres_checkpointer())

# Initialize the agent with the PostgreSQL checkpointer
agent = create_react_agent(
    model, 
    tools, 
    prompt=prompt,
    checkpointer=checkpointer
)