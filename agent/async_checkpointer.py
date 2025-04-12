import logging
import os
import dotenv
import urllib.parse
import asyncio
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
dotenv.load_dotenv()


async def get_async_checkpointer():
    """
    Creates and returns an asynchronous PostgreSQL checkpointer instance for use with LangGraph agents.

    Uses environment variables for database credentials.

    Returns:
        AsyncPostgresSaver: Configured asynchronous PostgreSQL checkpointer instance
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
    logger.info(f"DB Connection established to {DB_URI} with schema [{DB_SCHEMA}]")
    logger.info(f"DB Connection string: {db_connection_string}")

    connection_kwargs = {
        "autocommit": True,
        "prepare_threshold": 0,
    }

    async with AsyncConnectionPool(
        conninfo=DB_URI,
        max_size=20,
        kwargs=connection_kwargs,
    ) as pool:
        checkpointer = AsyncPostgresSaver(pool)

        try:
            await checkpointer.setup()
        except Exception as e:
            logger.error(f"Error setting up checkpointer: {e}")

        return checkpointer
