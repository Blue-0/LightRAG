"""Tests for entity extraction error handling - graceful chunk failure recovery."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from lightrag.exceptions import PipelineCancelledException
from lightrag.utils import Tokenizer, TokenizerInterface


class DummyTokenizer(TokenizerInterface):
    """Simple 1:1 character-to-token mapping for testing."""

    def encode(self, content: str):
        return [ord(ch) for ch in content]

    def decode(self, tokens):
        return "".join(chr(token) for token in tokens)


def _make_global_config(
    max_extract_input_tokens: int = 20480,
    entity_extract_max_gleaning: int = 0,
) -> dict:
    """Build a minimal global_config dict for extract_entities."""
    tokenizer = Tokenizer("dummy", DummyTokenizer())
    return {
        "llm_model_func": AsyncMock(return_value=""),
        "entity_extract_max_gleaning": entity_extract_max_gleaning,
        "addon_params": {},
        "tokenizer": tokenizer,
        "max_extract_input_tokens": max_extract_input_tokens,
        "llm_model_max_async": 4,
    }


# Minimal valid extraction result that _process_extraction_result can parse
_EXTRACTION_RESULT = (
    "(entity<|#|>TEST_ENTITY<|#|>CONCEPT<|#|>A test entity)<|COMPLETE|>"
)


def _make_chunks(count: int = 3) -> dict[str, dict]:
    """Create multiple test chunks."""
    chunks = {}
    for i in range(count):
        content = f"Test content for chunk {i}."
        chunks[f"chunk-{i:03d}"] = {
            "tokens": len(content),
            "content": content,
            "full_doc_id": "doc-001",
            "chunk_order_index": i,
        }
    return chunks


@pytest.mark.offline
@pytest.mark.asyncio
async def test_single_chunk_failure_does_not_abort_other_chunks():
    """When one chunk fails extraction, the remaining chunks should still be processed."""
    from lightrag.operate import extract_entities

    global_config = _make_global_config()

    call_count = 0

    async def _llm_side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        # Fail on the second call (chunk-001)
        if call_count == 2:
            raise RuntimeError("LLM API error for chunk")
        return _EXTRACTION_RESULT

    global_config["llm_model_func"] = AsyncMock(side_effect=_llm_side_effect)

    with patch("lightrag.operate.logger"):
        results = await extract_entities(
            chunks=_make_chunks(3),
            global_config=global_config,
        )

    # We had 3 chunks, 1 failed => 2 successful results
    assert len(results) == 2


@pytest.mark.offline
@pytest.mark.asyncio
async def test_all_chunks_fail_returns_empty():
    """When all chunks fail extraction, an empty list should be returned."""
    from lightrag.operate import extract_entities

    global_config = _make_global_config()
    global_config["llm_model_func"] = AsyncMock(
        side_effect=RuntimeError("LLM unavailable")
    )

    with patch("lightrag.operate.logger"):
        results = await extract_entities(
            chunks=_make_chunks(2),
            global_config=global_config,
        )

    assert results == []


@pytest.mark.offline
@pytest.mark.asyncio
async def test_cancellation_exception_still_propagates():
    """PipelineCancelledException should still abort all processing."""
    from lightrag.operate import extract_entities

    global_config = _make_global_config()
    # Set max_async to 1 so chunks are processed sequentially
    global_config["llm_model_max_async"] = 1

    call_count = 0

    async def _llm_side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _EXTRACTION_RESULT
        raise PipelineCancelledException("User cancelled")

    global_config["llm_model_func"] = AsyncMock(side_effect=_llm_side_effect)

    pipeline_status = {
        "cancellation_requested": False,
        "latest_message": "",
        "history_messages": [],
    }
    pipeline_status_lock = asyncio.Lock()

    with pytest.raises(PipelineCancelledException):
        with patch("lightrag.operate.logger"):
            await extract_entities(
                chunks=_make_chunks(3),
                global_config=global_config,
                pipeline_status=pipeline_status,
                pipeline_status_lock=pipeline_status_lock,
            )


@pytest.mark.offline
@pytest.mark.asyncio
async def test_failed_chunk_error_is_logged():
    """Failed chunk errors should be logged."""
    from lightrag.operate import extract_entities

    global_config = _make_global_config()

    call_count = 0

    async def _llm_side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("LLM timeout")
        return _EXTRACTION_RESULT

    global_config["llm_model_func"] = AsyncMock(side_effect=_llm_side_effect)

    with patch("lightrag.operate.logger") as mock_logger:
        await extract_entities(
            chunks=_make_chunks(2),
            global_config=global_config,
        )

    # Verify that the error was logged
    mock_logger.error.assert_called()
    error_calls = [str(c) for c in mock_logger.error.call_args_list]
    assert any("Chunk extraction failed" in c for c in error_calls)


@pytest.mark.offline
@pytest.mark.asyncio
async def test_pipeline_status_updated_on_chunk_failure():
    """Pipeline status should be updated when a chunk fails."""
    from lightrag.operate import extract_entities

    global_config = _make_global_config()

    call_count = 0

    async def _llm_side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("API error")
        return _EXTRACTION_RESULT

    global_config["llm_model_func"] = AsyncMock(side_effect=_llm_side_effect)

    pipeline_status = {
        "cancellation_requested": False,
        "latest_message": "",
        "history_messages": [],
    }
    pipeline_status_lock = asyncio.Lock()

    with patch("lightrag.operate.logger"):
        await extract_entities(
            chunks=_make_chunks(2),
            global_config=global_config,
            pipeline_status=pipeline_status,
            pipeline_status_lock=pipeline_status_lock,
        )

    # Verify pipeline status was updated with the error
    assert any(
        "Chunk extraction failed" in msg for msg in pipeline_status["history_messages"]
    )
