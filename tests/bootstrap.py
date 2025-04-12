import asyncio
from concurrent.futures import thread
import os
import dotenv
import urllib.parse
import logging
import psycopg
from uuid import uuid4
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from fake_agent import build_fake_agent_graph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
dotenv.load_dotenv()

connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": 0,
}

tools = [TavilySearchResults(max_results=1)]
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)



async def get_sync_db_connection(conn_string, **kwargs):
    """
    Get a connection to the PostgreSQL database.
    """
    logger.info("Creating synchronous connection to DB")
    return psycopg.connect(conn_string, **kwargs)


async def set_schema(conn, schema_name):
    """
    Set the search path to the specified schema in the PostgreSQL database.
    """
    logger.info(f"Setting schema to {schema_name}")
    try:
        with conn.cursor() as cur:
            cur.execute(f"SET search_path TO {schema_name};")
    except Exception as e:
        logger.error(f"Error setting schema: {e}")
        raise e


async def check_for_tables(conn):
    """
    Check if the necessary tables exist in the PostgreSQL database.
    """
    table_exists = False
    with conn.cursor() as cur:
        # Just check if the table existsm, don't create it
        cur.execute(
            f"""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'checkpoints'
            );
        """
        )
        table_exists = cur.fetchone()[0]
        if not table_exists:
            logger.debug("The 'checkpoints' table does not exist, setup will be called")
        else:
            logger.debug("The 'checkpoints' table exists, setup will be skipped")

        return table_exists


async def get_sync_checkpointer():
    """
    Get a synchronous PostgreSQL checkpointer instance. This is useful for testing locally.
    """

    DB_URI = os.environ.get("SUPABASE_DB_URI")
    DB_USERNAME = os.environ.get("SUPABASE_DB_USERNAME")
    DB_PASSWORD = os.environ.get("SUPABASE_DB_PASSWORD")
    DB_SCHEMA = os.environ.get("SUPABASE_DB_SCHEMA", "langgraph")

    # Encode the username and password for URI
    encoded_username = urllib.parse.quote(DB_USERNAME)
    encoded_password = urllib.parse.quote(DB_PASSWORD)

    db_connection_string = (
        f"postgresql://{encoded_username}:{encoded_password}@{DB_URI}?sslmode=disable"
    )

    logger.info(f"Creating direct database connection to {DB_URI}")
    conn = await get_sync_db_connection(db_connection_string, **connection_kwargs)

    logger.info(f"Connected to db, setting schema to {DB_SCHEMA}")
    await set_schema(conn, DB_SCHEMA)

    # Create the PostgresSaver instance
    logger.info("Creating PostgresSaver instance from connection")
    checkpointer = PostgresSaver(conn)

    # Check if the table exists
    logger.info("Schema set, checking if the 'checkpoints' table exists")
    tables_exist = await check_for_tables(conn)
    
    # Setup the checkpointer tables if they do not exist
    if not tables_exist:
        try:
            logger.info(f"Setting up the checkpoint tables in schema {DB_SCHEMA}")
            checkpointer.setup()
        except Exception as e:
            logger.error(f"Error setting up the checkpointer tables in schema {DB_SCHEMA}: {e}")

    return checkpointer

logger.info("Getting sync checkpointer")
checkpointer = asyncio.run(get_sync_checkpointer())

logger.info("Creating agent graph")
graph = build_fake_agent_graph(checkpointer=checkpointer)
#graph = create_react_agent(model, tools=tools, checkpointer=checkpointer)

thread_id = str(uuid4())
logger.info(f"Invoking agent graph with thread ID {thread_id}")
config = {"configurable": {"thread_id": thread_id}}
res = graph.invoke({"messages": [("human", "what's the weather in sf")]}, config)

checkpoint_tuples = list(checkpointer.list(config))
logger.info(f"Checkpoint tuples: {checkpoint_tuples}")