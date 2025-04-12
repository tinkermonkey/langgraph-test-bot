import unittest
import os
import urllib.parse
import logging
import uuid
import dotenv
import psycopg
import time
import asyncio
from agent.graph import agent

dotenv.load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)

class TestLangGraphAgent(unittest.TestCase):
    def setUp(self):
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

        # Database connection setup
        self.connection = psycopg.connect(db_connection_string)

        # Select the correct schema
        logger.info(f"Setting schema to {DB_SCHEMA}")
        try:
            with self.connection.cursor() as cur:
                cur.execute(f"SET search_path TO {DB_SCHEMA};")
        except Exception as e:
            logger.error(f"Error setting schema: {e}")
            raise e

        self.cursor = self.connection.cursor()

    def tearDown(self):
        # Clean up any database records from tests
        if hasattr(self, 'thread_id'):
            try:
                # Remove checkpoint data with the thread ID
                self.cursor.execute("DELETE FROM checkpoint_blobs WHERE thread_id = %s", (self.thread_id,))
                self.cursor.execute("DELETE FROM checkpoint_writes WHERE thread_id = %s", (self.thread_id,))
                self.cursor.execute("DELETE FROM checkpoints WHERE thread_id = %s", (self.thread_id,))
                self.connection.commit()
                logger.info(f"Cleaned up database records for thread {self.thread_id}")
            except Exception as e:
                logger.error(f"Error cleaning up database records: {e}")
        
        # Cleanup database connection
        self.cursor.close()
        self.connection.close()

    async def async_test_agent_checkpointing(self):
        # Define a test prompt
        test_prompt = "What is the capital of France?"

        # Run the agent with the test prompt
        self.thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": self.thread_id}}
        logger.info(f"Running test prompt on thread {self.thread_id}: {test_prompt}")
        response = await agent.invoke({"messages": [("human", test_prompt)]}, config=config)
        logger.debug(f"Response: {response}")

        # Check if the response is valid
        self.assertGreaterEqual(
            len(response['messages']), 2, "No response messages were returned."
        )
        response_content = response['messages'][-1].content
        logger.info(f"Response content: {response_content}")
        self.assertIn(
            "Paris", response_content, "Agent did not return the expected response."
        )

        max_wait_time = 5
        start_time = time.time()
        while time.time() - start_time < max_wait_time:
            # Check if the checkpoint exists in the database
            query = "SELECT COUNT(*) FROM checkpoints WHERE thread_id = %s"
            self.cursor.execute(query, (self.thread_id,))
            result = self.cursor.fetchone()
            logger.info(f"Checkpoint query result: {result[0]}")
            if result[0] >= 1:
                self.assertGreaterEqual(
                    result[0], 1, "Checkpoint for the test prompt was not created."
                )
                break
            time.sleep(1)
        else:
            logger.error("Checkpoint creation timed out.")
            self.fail("Checkpoint creation timed out.")

    def test_agent_checkpointing(self):
        asyncio.run(self.async_test_agent_checkpointing())


if __name__ == "__main__":
    unittest.main()
