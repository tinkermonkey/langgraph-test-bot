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

def test_checkpoint_multiple_threads(checkpointer):
    """
    Test saving and retrieving checkpoints for multiple threads
    """
    thread_configs = [
        {"configurable": {"thread_id": f"thread_{i}", "checkpoint_ns": checkpoint_ns}} 
        for i in range(3)
    ]

    saved_configs = []
    for config in thread_configs:
        checkpoint = {
            "v": 1,
            "id": f"checkpoint_{config['configurable']['thread_id']}",
            "ts": "2024-01-01T00:00:00+00:00",
            "channel_values": {"thread": config['configurable']['thread_id']}
        }

        saved_config = checkpointer.put(
            config, 
            checkpoint, 
            {"step": 1, "source": "test"}, 
            {}
        )
        saved_configs.append(saved_config)

    # Verify each thread's checkpoint can be retrieved
    for saved_config in saved_configs:
        retrieved_checkpoint = checkpointer.get(saved_config)
        assert retrieved_checkpoint is not None

def test_checkpoint_list_functionality(checkpointer):
    """
    Test listing checkpoints with various filters
    """
    # Save multiple checkpoints
    base_config = {"configurable": {"thread_id": "list_test_thread", "checkpoint_ns": checkpoint_ns}}

    for i in range(5):
        checkpoint = {
            "v": 1,
            "id": f"list_checkpoint_{i}",
            "ts": f"2024-01-0{i+1}T00:00:00+00:00",
            "channel_values": {"index": i}
        }
        checkpointer.put(
            base_config, 
            checkpoint, 
            {"step": i, "source": "list_test"}, 
            {}
        )

    # List all checkpoints for this thread
    checkpoint_tuples = list(checkpointer.list(base_config))

    # Verify total number of checkpoints
    assert len(checkpoint_tuples) == 5, f"Expected 5 checkpoints, got {len(checkpoint_tuples)}"

    # Verify checkpoints are sorted in descending order (newest first)
    checkpoint_ids = [ct.checkpoint['id'] for ct in checkpoint_tuples]
    expected_ids = [f"list_checkpoint_{i}" for i in range(4, -1, -1)]
    assert checkpoint_ids == expected_ids, f"Checkpoints not sorted correctly. Expected {expected_ids}, got {checkpoint_ids}"

    # Test limit parameter
    limited_checkpoints = list(checkpointer.list(base_config, limit=3))
    assert len(limited_checkpoints) == 3, f"Limit parameter not working correctly. Expected 3, got {len(limited_checkpoints)}"

    # Test before parameter
    # Get the third checkpoint to use as a 'before' reference
    before_config = checkpoint_tuples[2].config
    before_checkpoints = list(checkpointer.list(base_config, before=before_config))

    assert len(before_checkpoints) == 2, f"Before parameter not filtering checkpoints correctly. Expected 2, got {len(before_checkpoints)}"
    assert all(
        ct.checkpoint['id'] < checkpoint_tuples[2].checkpoint['id'] 
        for ct in before_checkpoints
    ), "Before parameter not working as expected. Checkpoints are not correctly filtered."

def test_checkpoint_error_handling(checkpointer):
    """
    Test error handling and edge cases
    """
    # Try to get checkpoint with invalid configuration
    invalid_config = {"configurable": {"thread_id": "non_existent_thread", "checkpoint_ns": checkpoint_ns}}
    
    # Retrieve should return None for non-existent thread
    retrieved_checkpoint = checkpointer.get(invalid_config)
    assert retrieved_checkpoint is None, "Should return None for non-existent thread"

    # Try to list checkpoints with invalid configuration
    invalid_list = list(checkpointer.list(invalid_config))
    assert len(invalid_list) == 0, "Should return empty list for non-existent thread"
