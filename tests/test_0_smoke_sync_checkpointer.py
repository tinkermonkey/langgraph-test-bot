import pytest
from agent.checkpointer import get_sync_checkpointer, delete_checkpoints

checkpoint_ns = __name__

@pytest.fixture
async def checkpointer():
    checkpointer = await get_sync_checkpointer()
    return checkpointer

@pytest.hookimpl(tryfirst=True, hookwrapper=True)
async def pytest_sessionfinish(session, exitstatus):
    """
    Hook to execute cleanup code after all tests are complete.
    """
    #yield  # Run all other hooks first
    await delete_checkpoints(checkpoint_ns=checkpoint_ns)

def test_checkpoint_basic_functionality(checkpointer):
    """
    Test basic checkpoint saving and retrieval functionality
    """
    # Prepare test configuration
    config = {
        "configurable": {
            "thread_id": "test_thread_1",
            "checkpoint_ns": checkpoint_ns
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
    saved_config = checkpointer.put(
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
