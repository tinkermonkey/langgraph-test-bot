import pytest
import asyncio
from agent.async_checkpointer import get_async_checkpointer

@pytest.fixture
def checkpointer():
    checkpointer = asyncio.run(get_async_checkpointer())
    return checkpointer

async def test_checkpoint_basic_functionality(checkpointer):
    """
    Test basic checkpoint saving and retrieval functionality
    """
    # Prepare test configuration
    config = {
        "configurable": {
            "thread_id": "test_thread_1",
            "checkpoint_ns": "test_namespace"
        }
    }

    # Sample checkpoint data
    checkpoint = {
        "v": 1,
        "id": "test_checkpoint_id",
        "ts": "2024-01-01T00:00:00+00:00",
        "channel_values": {"key": "value"}
    }

    # Sample metadata
    metadata = {
        "step": 1,
        "source": "test",
        "writes": {"channel": "test_write"}
    }

    # Simulate channel versions
    channel_versions = {
        "test_channel": "version_1"
    }

    # Save checkpoint
    saved_config = await checkpointer.aput(
        config, 
        checkpoint, 
        metadata, 
        channel_versions
    )

    # Verify saved configuration includes checkpoint ID
    assert "checkpoint_id" in saved_config["configurable"]
    assert saved_config["configurable"]["thread_id"] == "test_thread_1"

    # Retrieve the checkpoint
    retrieved_checkpoint = checkpointer.get(saved_config)
    assert retrieved_checkpoint is not None
    assert retrieved_checkpoint["id"] == checkpoint["id"]
