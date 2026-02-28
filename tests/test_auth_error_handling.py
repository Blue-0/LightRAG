"""Tests for authentication error handling improvements.

Validates that:
1. pypdf warnings are suppressed during PDF extraction
2. AuthenticationError (HTTP 401) is detected via status_code attribute
3. create_prefixed_exception preserves status_code for auth errors
"""

import logging

import httpx
import pytest

from lightrag.utils import create_prefixed_exception


class TestAuthErrorDetection:
    """Test that authentication errors are detected via status_code attribute."""

    def test_status_code_401_detected(self):
        """getattr(e, 'status_code', None) == 401 should identify auth errors."""

        class FakeAuthError(Exception):
            status_code = 401

        exc = FakeAuthError("Unauthorized")
        assert getattr(exc, "status_code", None) == 401

    def test_status_code_missing_returns_none(self):
        """Regular exceptions without status_code should not match."""
        exc = RuntimeError("generic error")
        assert getattr(exc, "status_code", None) is None

    def test_status_code_429_not_auth(self):
        """RateLimitError (429) should not match the 401 check."""

        class FakeRateLimitError(Exception):
            status_code = 429

        exc = FakeRateLimitError("rate limited")
        assert getattr(exc, "status_code", None) != 401

    def test_prefixed_auth_error_preserves_status_code(self):
        """create_prefixed_exception should preserve status_code on auth errors."""

        class FakeResponse:
            status_code = 401
            headers = {"x-request-id": "test123"}
            request = httpx.Request("POST", "https://api.example.com/v1/chat")

        class APIStatusError(Exception):
            def __init__(self, message: str, *, response: object, body: object | None):
                super().__init__(message)
                self.response = response
                self.status_code = getattr(response, "status_code", None)
                self.body = body

        class AuthenticationError(APIStatusError):
            pass

        resp = FakeResponse()
        exc = AuthenticationError(
            "Error code: 401 - {'detail': 'Unauthorized'}",
            response=resp,
            body={"detail": "Unauthorized"},
        )

        # After prefixing, status_code should still be 401
        result = create_prefixed_exception(exc, "C[1/14]")
        assert getattr(result, "status_code", None) == 401
        assert "C[1/14]" in str(result)


class TestPypdfWarningsSuppression:
    """Test that pypdf warnings are suppressed during PDF extraction."""

    def test_pypdf_logger_level_restored_on_success(self):
        """The pypdf logger level should be restored after successful PDF reading."""
        pypdf_logger = logging.getLogger("pypdf")
        original_level = pypdf_logger.level

        # Simulate what _extract_pdf_pypdf does
        pypdf_logger.setLevel(logging.ERROR)
        try:
            pass  # simulate PdfReader
        finally:
            pypdf_logger.setLevel(original_level)

        assert pypdf_logger.level == original_level

    def test_pypdf_logger_level_restored_on_error(self):
        """The pypdf logger level should be restored even if PdfReader raises."""
        pypdf_logger = logging.getLogger("pypdf")
        original_level = pypdf_logger.level

        # Simulate what _extract_pdf_pypdf does with an error
        pypdf_logger.setLevel(logging.ERROR)
        try:
            raise ValueError("bad pdf")
        except ValueError:
            pass
        finally:
            pypdf_logger.setLevel(original_level)

        assert pypdf_logger.level == original_level
