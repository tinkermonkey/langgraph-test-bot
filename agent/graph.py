import os
import dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

dotenv.load_dotenv(".env", override=True)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "langgraph-test-bot"

tools = [TavilySearchResults(max_results=1)]

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = (
    "You are a helpful assistant. "
    "You may not need to use tools for every query - the user may just want to chat!"
)

agent = create_react_agent(model, tools, prompt=prompt)