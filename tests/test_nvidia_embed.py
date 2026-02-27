"""Tests for NVIDIA asymmetric embedding model support in openai_embed."""

import base64
import struct
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest


@pytest.mark.offline
class TestNvidiaEmbedding:
    """Tests verifying NVIDIA-specific extra_body parameters are sent correctly."""

    def _make_mock_response(self, dim: int = 4, count: int = 1):
        """Create a mock embeddings response with base64-encoded vectors."""
        data_items = []
        for _ in range(count):
            vec = [0.1] * dim
            b64 = base64.b64encode(struct.pack(f"{dim}f", *vec)).decode()
            item = MagicMock()
            item.embedding = b64
            data_items.append(item)

        response = MagicMock()
        response.data = data_items
        response.usage = MagicMock(prompt_tokens=10, total_tokens=10)
        return response

    @pytest.mark.asyncio
    async def test_nvidia_model_sends_extra_body_passage(self):
        """Long texts should be sent with input_type='passage'."""
        from lightrag.llm.openai import openai_embed

        mock_response = self._make_mock_response(dim=4, count=1)
        mock_client = AsyncMock()
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        long_text = "A" * 200  # > 100 chars → passage

        with patch(
            "lightrag.llm.openai.create_openai_async_client",
            return_value=mock_client,
        ):
            result = await openai_embed.func(
                [long_text],
                model="nvidia/NV-Embed-v2",
                max_token_size=0,
            )

        actual_kwargs = mock_client.embeddings.create.call_args.kwargs
        extra_body = actual_kwargs.get("extra_body")
        assert extra_body is not None, "extra_body should be set for NVIDIA models"
        assert extra_body["input_type"] == "passage"
        assert extra_body["encoding_format"] == "float"
        assert extra_body["modality"] == ["text"]
        assert isinstance(result, np.ndarray)

    @pytest.mark.asyncio
    async def test_nvidia_model_sends_extra_body_query(self):
        """Short texts should be sent with input_type='query'."""
        from lightrag.llm.openai import openai_embed

        mock_response = self._make_mock_response(dim=4, count=1)
        mock_client = AsyncMock()
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        short_text = "Hello?"  # < 100 chars → query

        with patch(
            "lightrag.llm.openai.create_openai_async_client",
            return_value=mock_client,
        ):
            result = await openai_embed.func(
                [short_text],
                model="nvidia/NV-Embed-v2",
                max_token_size=0,
            )

        actual_kwargs = mock_client.embeddings.create.call_args.kwargs
        extra_body = actual_kwargs.get("extra_body")
        assert extra_body is not None
        assert extra_body["input_type"] == "query"

    @pytest.mark.asyncio
    async def test_non_nvidia_model_no_extra_body(self):
        """Non-NVIDIA models should NOT have extra_body."""
        from lightrag.llm.openai import openai_embed

        mock_response = self._make_mock_response(dim=4, count=1)
        mock_client = AsyncMock()
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "lightrag.llm.openai.create_openai_async_client",
            return_value=mock_client,
        ):
            await openai_embed.func(
                ["some text"],
                model="text-embedding-3-small",
                max_token_size=0,
            )

        actual_kwargs = mock_client.embeddings.create.call_args.kwargs
        assert "extra_body" not in actual_kwargs
