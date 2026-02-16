"""
Tests for the tmam.__init__ module (TmamConfig, init, trace decorator, etc.).
"""

import pytest
import os
from unittest.mock import MagicMock, patch, PropertyMock

import sys

sys.path.insert(0, "/home/claude/sdk/sdk/src")


class TestTmamConfig:
    """Tests for TmamConfig singleton class."""

    def test_singleton_pattern(self):
        """Test that TmamConfig is a singleton."""
        from tmam import TmamConfig

        config1 = TmamConfig()
        config2 = TmamConfig()
        assert config1 is config2

    def test_default_values(self):
        """Test default configuration values."""
        from tmam import TmamConfig

        TmamConfig.reset_to_defaults()

        assert TmamConfig.environment == "default"
        assert TmamConfig.application_name == "default"
        assert TmamConfig.pricing_info == {}
        assert TmamConfig.tracer is None
        assert TmamConfig.disable_batch is False
        assert TmamConfig.capture_message_content is True
        assert TmamConfig.disable_metrics is False

    def test_reset_to_defaults(self):
        """Test that reset_to_defaults properly resets all values."""
        from tmam import TmamConfig

        # Set some values
        TmamConfig.environment = "production"
        TmamConfig.application_name = "test-app"
        TmamConfig.disable_batch = True

        # Reset
        TmamConfig.reset_to_defaults()

        assert TmamConfig.environment == "default"
        assert TmamConfig.application_name == "default"
        assert TmamConfig.disable_batch is False

    @patch("tmam.fetch_pricing_info")
    def test_update_config(self, mock_fetch_pricing):
        """Test update_config method."""
        from tmam import TmamConfig

        mock_fetch_pricing.return_value = {"chat": {}}
        mock_tracer = MagicMock()
        mock_event_provider = MagicMock()
        mock_metrics_dict = {"counter": MagicMock()}

        TmamConfig.update_config(
            environment="production",
            application_name="my-app",
            tracer=mock_tracer,
            event_provider=mock_event_provider,
            disable_batch=True,
            capture_message_content=False,
            metrics_dict=mock_metrics_dict,
            disable_metrics=True,
            pricing_json=None,
            url="https://api.test.com",
            public_key="pk_test",
            secrect_key="sk_test",
            last_guard_prompt_id=None,
            guardrail_id="guardrail-123",
            name=None,
            user_id=None,
        )

        assert TmamConfig.environment == "production"
        assert TmamConfig.application_name == "my-app"
        assert TmamConfig.tracer is mock_tracer
        assert TmamConfig.disable_batch is True
        assert TmamConfig.capture_message_content is False
        assert TmamConfig.guardrail_id == "guardrail-123"

    def test_update_guard_config(self):
        """Test update_guard_config method."""
        from tmam import TmamConfig

        TmamConfig.reset_to_defaults()

        TmamConfig.update_guard_config(
            last_guard_prompt_id="prompt-123",
            name="test-guard",
            user_id="user-456",
            guardrail_id="guardrail-789",
        )

        assert TmamConfig.last_guard_prompt_id == "prompt-123"
        assert TmamConfig.name == "test-guard"
        assert TmamConfig.user_id == "user-456"
        assert TmamConfig.guardrail_id == "guardrail-789"


class TestModuleExists:
    """Tests for module_exists function."""

    def test_existing_module(self):
        """Test detection of existing module."""
        from tmam import module_exists

        assert module_exists("os") is True
        assert module_exists("sys") is True
        assert module_exists("json") is True

    def test_nonexistent_module(self):
        """Test detection of nonexistent module."""
        from tmam import module_exists

        assert module_exists("nonexistent_module_xyz") is False

    def test_nested_module(self):
        """Test detection of nested module."""
        from tmam import module_exists

        assert module_exists("os.path") is True
        assert module_exists("xml.etree.ElementTree") is True


class TestInstrumentIfAvailable:
    """Tests for instrument_if_available function."""

    def test_disabled_instrumentor_skipped(self):
        """Test that disabled instrumentors are skipped."""
        from tmam import instrument_if_available, TmamConfig

        mock_instrumentor = MagicMock()
        config = TmamConfig()

        instrument_if_available(
            "openai",
            mock_instrumentor,
            config,
            disabled_instrumentors=["openai"],
            module_name_map={"openai": "openai"},
        )

        mock_instrumentor.instrument.assert_not_called()

    def test_missing_module_mapping(self):
        """Test handling of missing module mapping."""
        from tmam import instrument_if_available, TmamConfig

        mock_instrumentor = MagicMock()
        config = TmamConfig()

        # Should not raise, but also not instrument
        instrument_if_available(
            "unknown_provider",
            mock_instrumentor,
            config,
            disabled_instrumentors=[],
            module_name_map={},
        )

        mock_instrumentor.instrument.assert_not_called()


class TestTraceDecorator:
    """Tests for trace decorator."""

    @patch("tmam.t.get_tracer_provider")
    def test_trace_wraps_function(self, mock_get_tracer_provider):
        """Test that trace decorator wraps function."""
        from tmam import trace

        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_span.__enter__ = MagicMock(return_value=mock_span)
        mock_span.__exit__ = MagicMock(return_value=False)
        mock_tracer.start_as_current_span.return_value = mock_span
        mock_get_tracer_provider.return_value.get_tracer.return_value = mock_tracer

        @trace
        def test_function(x, y):
            return x + y

        result = test_function(1, 2)

        assert result == 3
        mock_tracer.start_as_current_span.assert_called_once()

    def test_trace_raises_on_non_callable(self):
        """Test that trace raises TypeError for non-callable."""
        from tmam import trace

        with pytest.raises(TypeError) as exc_info:
            trace("not a function")

        assert "callable objects" in str(exc_info.value)

    @patch("tmam.t.get_tracer_provider")
    def test_trace_records_exception(self, mock_get_tracer_provider):
        """Test that trace decorator records exceptions."""
        from tmam import trace

        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_span.__enter__ = MagicMock(return_value=mock_span)
        mock_span.__exit__ = MagicMock(return_value=False)
        mock_tracer.start_as_current_span.return_value = mock_span
        mock_get_tracer_provider.return_value.get_tracer.return_value = mock_tracer

        @trace
        def failing_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            failing_function()

        mock_span.record_exception.assert_called_once()


class TestTracedSpan:
    """Tests for TracedSpan class."""

    def test_set_result(self, mock_span):
        """Test set_result method."""
        from tmam import TracedSpan

        traced = TracedSpan(mock_span)
        traced.set_result("test result")

        mock_span.set_attribute.assert_called_once()
        args = mock_span.set_attribute.call_args
        assert "test result" in str(args)

    def test_set_metadata(self, mock_span):
        """Test set_metadata method."""
        from tmam import TracedSpan

        traced = TracedSpan(mock_span)
        metadata = {"key1": "value1", "key2": "value2"}
        traced.set_metadata(metadata)

        mock_span.set_attributes.assert_called_once_with(attributes=metadata)

    def test_context_manager(self, mock_span):
        """Test context manager protocol."""
        from tmam import TracedSpan

        traced = TracedSpan(mock_span)

        with traced as span:
            assert span is traced

        mock_span.end.assert_called_once()


class TestStartTrace:
    """Tests for start_trace context manager."""

    @patch("tmam.t.get_tracer_provider")
    def test_yields_traced_span(self, mock_get_tracer_provider):
        """Test that start_trace yields a TracedSpan."""
        from tmam import start_trace, TracedSpan

        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_span.__enter__ = MagicMock(return_value=mock_span)
        mock_span.__exit__ = MagicMock(return_value=False)
        mock_tracer.start_as_current_span.return_value = mock_span
        mock_get_tracer_provider.return_value.get_tracer.return_value = mock_tracer

        with start_trace("test-span") as traced:
            assert isinstance(traced, TracedSpan)


class TestGetPrompt:
    """Tests for get_prompt function."""

    def test_raises_without_init(self):
        """Test that get_prompt raises when tmam not initialized."""
        from tmam import get_prompt, TmamConfig

        TmamConfig.reset_to_defaults()
        TmamConfig.url = None
        TmamConfig.public_key = None
        TmamConfig.secrect_key = None

        with pytest.raises(ValueError) as exc_info:
            get_prompt(name="test-prompt")

        assert "tmam.init" in str(exc_info.value)

    @patch("requests.post")
    def test_successful_prompt_fetch(self, mock_post):
        """Test successful prompt fetch."""
        from tmam import get_prompt, TmamConfig

        TmamConfig.url = "https://api.test.com/v1"
        TmamConfig.public_key = "pk_test"
        TmamConfig.secrect_key = "sk_test"

        mock_response = MagicMock()
        mock_response.json.return_value = {"data": {"template": "Hello {{name}}"}}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        result = get_prompt(name="greeting")

        assert result == {"template": "Hello {{name}}"}
        mock_post.assert_called_once()

    @patch("requests.post")
    def test_failed_prompt_fetch(self, mock_post):
        """Test handling of failed prompt fetch."""
        from tmam import get_prompt, TmamConfig
        import requests

        TmamConfig.url = "https://api.test.com/v1"
        TmamConfig.public_key = "pk_test"
        TmamConfig.secrect_key = "sk_test"

        mock_post.side_effect = requests.RequestException("Connection failed")

        result = get_prompt(name="greeting")

        assert result is None


class TestGetSecrets:
    """Tests for get_secrets function."""

    def test_raises_without_init(self):
        """Test that get_secrets raises when tmam not initialized."""
        from tmam import get_secrets, TmamConfig

        TmamConfig.reset_to_defaults()
        TmamConfig.url = None

        with pytest.raises(ValueError) as exc_info:
            get_secrets(key="test-key")

        assert "tmam.init" in str(exc_info.value)

    @patch("requests.post")
    def test_successful_secrets_fetch(self, mock_post):
        """Test successful secrets fetch."""
        from tmam import get_secrets, TmamConfig

        TmamConfig.url = "https://api.test.com/v1"
        TmamConfig.public_key = "pk_test"
        TmamConfig.secrect_key = "sk_test"

        mock_response = MagicMock()
        mock_response.json.return_value = {"data": {"res": {"API_KEY": "secret123"}}}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        result = get_secrets(key="API_KEY")

        assert result == {"res": {"API_KEY": "secret123"}}

    @patch("requests.post")
    def test_secrets_set_env(self, mock_post):
        """Test that secrets are set as environment variables when requested."""
        from tmam import get_secrets, TmamConfig

        TmamConfig.url = "https://api.test.com/v1"
        TmamConfig.public_key = "pk_test"
        TmamConfig.secrect_key = "sk_test"

        mock_response = MagicMock()
        mock_response.json.return_value = {"data": {"res": {"TEST_SECRET": "value123"}}}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        get_secrets(key="TEST_SECRET", should_set_env=True)

        assert os.environ.get("TEST_SECRET") == "value123"
        del os.environ["TEST_SECRET"]


class TestDetect:
    """Tests for Detect class."""

    def test_input_raises_without_init(self):
        """Test that detect.input raises when tmam not initialized."""
        from tmam import Detect, TmamConfig

        TmamConfig.reset_to_defaults()
        TmamConfig.url = None

        detect = Detect()

        with pytest.raises(ValueError) as exc_info:
            detect.input("test text")

        assert "tmam.init" in str(exc_info.value)

    @patch("requests.post")
    def test_input_successful(self, mock_post):
        """Test successful input detection."""
        from tmam import Detect, TmamConfig

        TmamConfig.url = "https://api.test.com/v1"
        TmamConfig.public_key = "pk_test"
        TmamConfig.secrect_key = "sk_test"
        TmamConfig.guardrail_id = "guardrail-123"

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": {
                "result": {
                    "verdict": "safe",
                    "score": 0.1,
                },
                "guardPromptId": "prompt-123",
            }
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        detect = Detect()
        result = detect.input("Hello, world!")

        assert result is not None
        mock_post.assert_called_once()
