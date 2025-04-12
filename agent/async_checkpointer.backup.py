import os
import dotenv
import urllib.parse
import asyncio
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
from eleanor.utils import get_logger

logger = get_logger(__name__)

dotenv.load_dotenv()
DB_URI = os.environ.get("SUPABASE_DB_URI")
DB_USERNAME = os.environ.get("SUPABASE_DB_USERNAME")
DB_PASSWORD = os.environ.get("SUPABASE_DB_PASSWORD")

# Encode the username and password for URI
encoded_username = urllib.parse.quote(DB_USERNAME)
encoded_password = urllib.parse.quote(DB_PASSWORD)

DB_URI = f"postgresql://{encoded_username}:{encoded_password}@{DB_URI}?sslmode=disable"

connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": 0,
}

async def create_checkpointer():
    async with AsyncConnectionPool(
        conninfo=DB_URI,
        max_size=20,
        kwargs=connection_kwargs,
    ) as pool:
        checkpointer = AsyncPostgresSaver(pool)

        return checkpointer

async def setup_checkpointer():
    """Set up the checkpointer and return it."""
    return await create_checkpointer()

# Run the setup function and export the checkpointer
checkpointer = asyncio.run(setup_checkpointer())