import pytest
from agent.checkpointer import get_sync_checkpointer

@pytest.fixture
async def checkpointer():
    checkpointer = await get_sync_checkpointer()
    return checkpointer

def test_checkpoint_basic_functionality(checkpointer):
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
        {"configurable": {"thread_id": f"thread_{i}", "checkpoint_ns": "test_namespace"}} 
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
    base_config = {"configurable": {"thread_id": "list_test_thread", "checkpoint_ns": "test_namespace"}}
    
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
    assert checkpoint_ids == [f"list_checkpoint_{i}" for i in range(4, -1, -1)], "Checkpoints not sorted correctly"

    # Test limit parameter
    limited_checkpoints = list(checkpointer.list(base_config, limit=3))
    assert len(limited_checkpoints) == 3, "Limit parameter not working correctly"

    # Test before parameter
    # Get the third checkpoint to use as a 'before' reference
    before_config = checkpoint_tuples[[2]]('https://docs.smith.langchain.com/reference/js/classes/vitest_reporter.default.html').config
    before_checkpoints = list(checkpointer.list(base_config, before=before_config))
    
    assert len(before_checkpoints) == 3, "Before parameter not filtering checkpoints correctly"
    assert all(
        ct.checkpoint['id'] > before_config['configurable']['checkpoint_id'] 
        for ct in before_checkpoints
    ), "Before parameter not working as expected"

def test_checkpoint_error_handling(checkpointer):
    """
    Test error handling and edge cases
    """
    # Try to get checkpoint with invalid configuration
    invalid_config = {"configurable": {"thread_id": "non_existent_thread", "checkpoint_ns": "test_namespace"}}
    
    # Retrieve should return None for non-existent thread
    retrieved_checkpoint = checkpointer.get(invalid_config)
    assert retrieved_checkpoint is None, "Should return None for non-existent thread"

    # Try to list checkpoints with invalid configuration
    invalid_list = list(checkpointer.list(invalid_config))
    assert len(invalid_list) == 0, "Should return empty list for non-existent thread"

def test_checkpoint_pending_writes(checkpointer):
    """
    Test handling of pending writes
    """
    config = {"configurable": {"thread_id": "pending_writes_test", "checkpoint_ns": "test_namespace"}}
    
    # Create a checkpoint with pending writes
    checkpoint = {
        "v": 1,
        "id": "pending_writes_checkpoint",
        "ts": "2024-01-01T00:00:00+00:00",
        "channel_values": {"key": "value"},
        "pending_sends": [{"task_id": "task1", "data": "write_data"}]
    }

    # Save checkpoint with pending writes
    saved_config = checkpointer.put(
        config, 
        checkpoint, 
        {"step": 1, "source": "pending_writes_test"}, 
        {}
    )

    # Retrieve the checkpoint
    retrieved_checkpoint = checkpointer.get(saved_config)
    
    # Verify pending writes are preserved
    assert "pending_sends" in retrieved_checkpoint, "Pending writes not preserved"
    assert len(retrieved_checkpoint.get("pending_sends", [])) > 0, "Pending writes list is empty"
