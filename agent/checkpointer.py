import os
import dotenv
import urllib.parse
import logging
import psycopg
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import AsyncConnectionPool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
dotenv.load_dotenv()

connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": 0,
}


def get_db_connection_string():
    """
    Get the database connection string from environment variables.
    """
    DB_URI = os.environ.get("SUPABASE_DB_URI")
    DB_USERNAME = os.environ.get("SUPABASE_DB_USERNAME")
    DB_PASSWORD = os.environ.get("SUPABASE_DB_PASSWORD")

    # Encode the username and password for URI
    encoded_username = urllib.parse.quote(DB_USERNAME)
    encoded_password = urllib.parse.quote(DB_PASSWORD)

    return (
        f"postgresql://{encoded_username}:{encoded_password}@{DB_URI}?sslmode=disable"
    )


async def get_db_connection(conn_string, **kwargs):
    """
    Get a connection to the PostgreSQL database.
    """
    logger.info("Creating synchronous connection to DB")
    return psycopg.connect(conn_string, **kwargs)


async def aget_db_connection(conn_string, **kwargs):
    """
    Get an asynchronous connection to the PostgreSQL database.
    """
    logger.info(f"Creating async connection to DB: {conn_string}")
    async with AsyncConnectionPool(conninfo=conn_string, max_size=15, **kwargs) as pool:
        logger.info("Async connection pool created")
        return pool


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
    DB_SCHEMA = os.environ.get("SUPABASE_DB_SCHEMA", "langgraph")

    logger.info(f"Creating direct database connection to {DB_URI}")
    db_connection_string = get_db_connection_string()
    conn = await get_db_connection(db_connection_string, **connection_kwargs)

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
            logger.error(
                f"Error setting up the checkpointer tables in schema {DB_SCHEMA}: {e}"
            )

    return checkpointer


async def delete_checkpoints(thread_id="", checkpoint_ns="", **connection_kwargs):
    """
    Delete checkpoints from the database for a specific thread ID and/or namespace.
    """
    DB_SCHEMA = os.environ.get("SUPABASE_DB_SCHEMA", "langgraph")

    if not thread_id and not checkpoint_ns:
        raise ValueError(
            "either thread_id or checkpoint_ns is required to delete checkpoints."
        )

    db_connection_string = get_db_connection_string()
    conn = await get_db_connection(db_connection_string, **connection_kwargs)

    logger.info(f"Connected to db, setting schema to {DB_SCHEMA}")
    await set_schema(conn, DB_SCHEMA)

    # use the connection to delete checkpoints
    table_names = ["checkpoint_blobs", "checkpoint_writes", "checkpoints"]
    with conn.cursor() as cur:
        for table_name in table_names:
            if thread_id:
                cur.execute(
                    f"DELETE FROM {table_name} WHERE thread_id = '%s'", (thread_id,)
                )
            if checkpoint_ns:
                cur.execute(
                    f"DELETE FROM {table_name} WHERE checkpoint_ns = '%s'",
                    (checkpoint_ns,),
                )
            conn.commit()
        logger.info(
            f"Deleted checkpoints for thread_id: {thread_id}, checkpoint_ns: {checkpoint_ns}"
        )

    return True
