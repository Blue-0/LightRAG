"""Tests for create_prefixed_exception in lightrag.utils."""

import httpx

from lightrag.utils import create_prefixed_exception


class TestCreatePrefixedException:
    """Test create_prefixed_exception with various exception types."""

    def test_basic_exception(self):
        """Method 1: basic Exception reconstruction."""
        exc = ValueError("something went wrong")
        result = create_prefixed_exception(exc, "PREFIX")
        assert isinstance(result, ValueError)
        assert "PREFIX: something went wrong" in str(result)

    def test_oserror_with_errno(self):
        """Method 1: OSError where first arg is an integer (errno)."""
        exc = OSError(2, "No such file or directory")
        result = create_prefixed_exception(exc, "PREFIX")
        assert isinstance(result, OSError)
        assert "PREFIX" in str(result)

    def test_exception_with_no_args(self):
        """Method 1: exception with no args."""
        exc = RuntimeError()
        result = create_prefixed_exception(exc, "PREFIX")
        assert isinstance(result, RuntimeError)
        assert "PREFIX" in str(result)

    def test_keyword_only_constructor(self):
        """Method 2: exception whose constructor requires keyword-only args.

        This simulates OpenAI's APIStatusError which needs
        ``__init__(message, *, response, body)`` and cannot be reconstructed
        with positional args alone.
        """

        class KWOnlyError(Exception):
            """Exception that requires keyword-only constructor args."""

            def __init__(self, message: str, *, response: object, body: object):
                super().__init__(message)
                self.response = response
                self.body = body

        exc = KWOnlyError("unauthorized", response="resp_obj", body={"detail": "Unauthorized"})
        result = create_prefixed_exception(exc, "C[1/14]")

        # Must preserve the original exception type
        assert isinstance(result, KWOnlyError)
        # The prefix must appear in the string representation
        assert "C[1/14]" in str(result)
        assert "unauthorized" in str(result)
        # Original attributes must survive
        assert result.response == "resp_obj"
        assert result.body == {"detail": "Unauthorized"}

    def test_keyword_only_constructor_is_same_object(self):
        """Method 2 returns the *same* exception object (mutated in-place)."""

        class KWOnlyError(Exception):
            def __init__(self, msg: str, *, extra: int):
                super().__init__(msg)
                self.extra = extra

        exc = KWOnlyError("boom", extra=42)
        result = create_prefixed_exception(exc, "PFX")
        assert result is exc  # same object, not a copy

    def test_openai_api_status_error_like(self):
        """Simulate the real-world OpenAI APIStatusError pattern.

        OpenAI's ``APIStatusError.__init__(message, *, response, body)``
        stores ``response`` and ``body`` as attributes and calls
        ``super().__init__(message)``.  Reconstruction via ``type(e)(*e.args)``
        fails because the keyword-only args are missing.
        """

        class FakeResponse:
            status_code = 401
            headers = {"x-request-id": "abc123"}
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

        result = create_prefixed_exception(exc, "chunk-abc123")

        assert isinstance(result, AuthenticationError)
        assert isinstance(result, APIStatusError)
        assert "chunk-abc123" in str(result)
        assert "401" in str(result)
        assert result.status_code == 401
        assert result.body == {"detail": "Unauthorized"}

    def test_method3_fallback_to_runtime_error(self):
        """Method 3: if both reconstruction and in-place mutation fail."""

        class ImmutableArgsError(Exception):
            """Exception where args cannot be set."""

            @property
            def args(self):
                return ("immutable",)

            @args.setter
            def args(self, value):
                raise AttributeError("cannot set args")

            def __init__(self):
                # Don't call super().__init__() to avoid setting args normally
                pass

        exc = ImmutableArgsError()
        result = create_prefixed_exception(exc, "PFX")
        assert isinstance(result, RuntimeError)
        assert "PFX" in str(result)
