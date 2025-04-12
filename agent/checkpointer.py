import os
import urllib.parse
import logging
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def get_postgres_checkpointer():
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
    
    db_connection_string = f"postgresql://{encoded_username}:{encoded_password}@{DB_URI}?sslmode=disable"
    logger.info(f"DB Connection established to {DB_URI} with schema [{DB_SCHEMA}]")
    logger.info(f"DB Connection string: {db_connection_string}")
    
    connection_kwargs = {
        "autocommit": True,
        "prepare_threshold": 0,
    }
    
    logger.info("Creating connection pool")
    async with AsyncConnectionPool(conninfo=db_connection_string, max_size=20, kwargs=connection_kwargs) as pool:
        logger.info("Connection pool created successfully, getting connection")
        async with pool.connection() as conn:
            logger.info("Connection acquired from pool, getting a cursor")
            async with conn.cursor() as cur:
                logger.info(f"Cursor acquired, setting schema to {DB_SCHEMA}")
                try:
                    await cur.execute(f"SET search_path TO {DB_SCHEMA};")
                except Exception as e:
                    logger.error(f"Error setting schema: {e}")
                    raise e

                # Check if the table exists
                logger.info("Schema set, checking if the 'checkpoints' table exists")
                table_exists = False
                await cur.execute(f"""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'checkpoints'
                    );
                """)
                table_exists = (await cur.fetchone())[0]
                if not table_exists:
                    logger.info("The 'checkpoints' table does not exist, setup will be called")
                else:
                    logger.info("The 'checkpoints' table exists, setup will be skipped")

        # Create the AsyncPostgresSaver instance
        logger.info("Creating AsyncPostgresSaver instance with the connection pool")
        checkpointer = AsyncPostgresSaver(pool)

        # Make sure the tables exist
        if not table_exists:
            try:
                logger.info("Setting up the checkpoint tables")
                await checkpointer.setup()
            except Exception as e:
                logger.error(f"Error setting up the database table: {e}")
                raise e

    logger.info("AsyncPostgresSaver instance created and setup completed")
    return checkpointer