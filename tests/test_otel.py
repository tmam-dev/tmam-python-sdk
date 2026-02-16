"""
Tests for the tmam.otel modules (tracing, metrics, events).
"""

import pytest
import os
from unittest.mock import MagicMock, patch, PropertyMock

import sys

sys.path.insert(0, "/home/claude/sdk/sdk/src")


class TestSetupTracing:
    """Tests for setup_tracing function."""

    def test_returns_external_tracer_if_provided(self):
        """Test that external tracer is returned if provided."""
        from tmam.otel.tracing import setup_tracing

        external_tracer = MagicMock()
        result = setup_tracing(
            application_name="test-app",
            environment="test",
            tracer=external_tracer,
            url=None,
            public_key=None,
            secrect_key=None,
            disable_batch=False,
        )

        assert result is external_tracer

    @patch("tmam.otel.tracing.trace")
    @patch("tmam.otel.tracing.TracerProvider")
    @patch("tmam.otel.tracing.OTLPSpanExporter")
    @patch("tmam.otel.tracing.BatchSpanProcessor")
    @patch("tmam.otel.tracing.TRACER_SET", False)
    def test_creates_tracer_provider(
        self, mock_batch, mock_exporter, mock_provider, mock_trace
    ):
        """Test that tracer provider is created."""
        from tmam.otel import tracing

        tracing.TRACER_SET = False

        mock_tracer = MagicMock()
        mock_trace.get_tracer_provider.return_value.get_tracer.return_value = (
            mock_tracer
        )
        mock_trace.get_tracer_provider.return_value.add_span_processor = MagicMock()

        result = tracing.setup_tracing(
            application_name="test-app",
            environment="test",
            tracer=None,
            url="https://api.test.com",
            public_key="pk_test",
            secrect_key="sk_test",
            disable_batch=False,
        )

        mock_provider.assert_called_once()
        mock_trace.set_tracer_provider.assert_called_once()

    @patch("tmam.otel.tracing.trace")
    @patch("tmam.otel.tracing.TracerProvider")
    @patch("tmam.otel.tracing.OTLPSpanExporter")
    @patch("tmam.otel.tracing.SimpleSpanProcessor")
    @patch("tmam.otel.tracing.TRACER_SET", False)
    def test_uses_simple_processor_when_batch_disabled(
        self, mock_simple, mock_exporter, mock_provider, mock_trace
    ):
        """Test that SimpleSpanProcessor is used when batch is disabled."""
        from tmam.otel import tracing

        tracing.TRACER_SET = False

        mock_tracer = MagicMock()
        mock_trace.get_tracer_provider.return_value.get_tracer.return_value = (
            mock_tracer
        )
        mock_trace.get_tracer_provider.return_value.add_span_processor = MagicMock()

        result = tracing.setup_tracing(
            application_name="test-app",
            environment="test",
            tracer=None,
            url="https://api.test.com",
            public_key="pk_test",
            secrect_key="sk_test",
            disable_batch=True,
        )

        mock_simple.assert_called_once()

    def test_sets_haystack_env_variable(self):
        """Test that HAYSTACK_AUTO_TRACE_ENABLED is set to false."""
        from tmam.otel import tracing

        # Clear the flag to allow setup
        tracing.TRACER_SET = False

        with patch("tmam.otel.tracing.trace"), patch(
            "tmam.otel.tracing.TracerProvider"
        ), patch("tmam.otel.tracing.OTLPSpanExporter"), patch(
            "tmam.otel.tracing.BatchSpanProcessor"
        ):

            tracing.setup_tracing(
                application_name="test-app",
                environment="test",
                tracer=None,
                url="https://api.test.com",
                public_key="pk_test",
                secrect_key="sk_test",
                disable_batch=False,
            )

        assert os.environ.get("HAYSTACK_AUTO_TRACE_ENABLED") == "false"


class TestSetupMeter:
    """Tests for setup_meter function."""

    @patch("tmam.otel.metrics.metrics")
    @patch("tmam.otel.metrics.MeterProvider")
    @patch("tmam.otel.metrics.OTLPMetricExporter")
    @patch("tmam.otel.metrics.PeriodicExportingMetricReader")
    @patch("tmam.otel.metrics.METER_SET", False)
    def test_creates_meter_provider(
        self, mock_reader, mock_exporter, mock_provider, mock_metrics
    ):
        """Test that meter provider is created."""
        from tmam.otel import metrics as metrics_module

        metrics_module.METER_SET = False

        mock_meter = MagicMock()
        mock_meter.create_histogram = MagicMock(return_value=MagicMock())
        mock_meter.create_counter = MagicMock(return_value=MagicMock())
        mock_metrics.get_meter.return_value = mock_meter

        result, err = metrics_module.setup_meter(
            application_name="test-app",
            environment="test",
            meter=None,
            url="https://api.test.com",
            public_key="pk_test",
            secrect_key="sk_test",
        )

        assert err is None
        assert result is not None
        mock_provider.assert_called_once()

    @patch("tmam.otel.metrics.metrics")
    @patch("tmam.otel.metrics.MeterProvider")
    @patch("tmam.otel.metrics.OTLPMetricExporter")
    @patch("tmam.otel.metrics.PeriodicExportingMetricReader")
    @patch("tmam.otel.metrics.METER_SET", False)
    def test_creates_all_metrics(
        self, mock_reader, mock_exporter, mock_provider, mock_metrics
    ):
        """Test that all expected metrics are created."""
        from tmam.otel import metrics as metrics_module

        metrics_module.METER_SET = False

        mock_meter = MagicMock()
        mock_histogram = MagicMock()
        mock_counter = MagicMock()
        mock_meter.create_histogram = MagicMock(return_value=mock_histogram)
        mock_meter.create_counter = MagicMock(return_value=mock_counter)
        mock_metrics.get_meter.return_value = mock_meter

        result, err = metrics_module.setup_meter(
            application_name="test-app",
            environment="test",
            meter=None,
            url="https://api.test.com",
            public_key="pk_test",
            secrect_key="sk_test",
        )

        assert err is None

        # Check for expected metric keys
        expected_keys = [
            "genai_client_usage_tokens",
            "genai_client_operation_duration",
            "genai_server_tbt",
            "genai_server_ttft",
            "db_client_operation_duration",
            "genai_requests",
            "genai_prompt_tokens",
            "genai_completion_tokens",
            "genai_cost",
            "db_requests",
        ]

        for key in expected_keys:
            assert key in result, f"Expected metric key '{key}' not found"

    @patch("tmam.otel.metrics.metrics")
    @patch("tmam.otel.metrics.MeterProvider")
    def test_handles_exception(self, mock_provider, mock_metrics):
        """Test that exceptions are handled gracefully."""
        from tmam.otel import metrics as metrics_module

        metrics_module.METER_SET = False

        mock_provider.side_effect = Exception("Test error")

        result, err = metrics_module.setup_meter(
            application_name="test-app",
            environment="test",
            meter=None,
            url="https://api.test.com",
            public_key="pk_test",
            secrect_key="sk_test",
        )

        assert result is None
        assert err is not None


class TestSetupEvents:
    """Tests for setup_events function."""

    def test_returns_external_event_logger_if_provided(self):
        """Test that external event logger is returned if provided."""
        from tmam.otel.events import setup_events

        external_logger = MagicMock()
        result = setup_events(
            application_name="test-app",
            environment="test",
            event_logger=external_logger,
            url=None,
            public_key=None,
            secrect_key=None,
            disable_batch=False,
        )

        assert result is external_logger

    @patch("tmam.otel.events._events")
    @patch("tmam.otel.events._logs")
    @patch("tmam.otel.events.LoggerProvider")
    @patch("tmam.otel.events.EventLoggerProvider")
    @patch("tmam.otel.events.OTLPLogExporter")
    @patch("tmam.otel.events.BatchLogRecordProcessor")
    @patch("tmam.otel.events.EVENTS_SET", False)
    def test_creates_event_logger_provider(
        self,
        mock_batch,
        mock_exporter,
        mock_event_provider,
        mock_logger_provider,
        mock_logs,
        mock_events,
    ):
        """Test that event logger provider is created."""
        from tmam.otel import events

        events.EVENTS_SET = False

        mock_event_logger = MagicMock()
        mock_events.get_event_logger.return_value = mock_event_logger

        result = events.setup_events(
            application_name="test-app",
            environment="test",
            event_logger=None,
            url="https://api.test.com",
            public_key="pk_test",
            secrect_key="sk_test",
            disable_batch=False,
        )

        mock_logger_provider.assert_called_once()
        mock_event_provider.assert_called_once()

    @patch("tmam.otel.events._events")
    @patch("tmam.otel.events._logs")
    @patch("tmam.otel.events.LoggerProvider")
    @patch("tmam.otel.events.EventLoggerProvider")
    @patch("tmam.otel.events.OTLPLogExporter")
    @patch("tmam.otel.events.SimpleLogRecordProcessor")
    @patch("tmam.otel.events.EVENTS_SET", False)
    def test_uses_simple_processor_when_batch_disabled(
        self,
        mock_simple,
        mock_exporter,
        mock_event_provider,
        mock_logger_provider,
        mock_logs,
        mock_events,
    ):
        """Test that SimpleLogRecordProcessor is used when batch is disabled."""
        from tmam.otel import events

        events.EVENTS_SET = False

        mock_event_logger = MagicMock()
        mock_events.get_event_logger.return_value = mock_event_logger

        result = events.setup_events(
            application_name="test-app",
            environment="test",
            event_logger=None,
            url="https://api.test.com",
            public_key="pk_test",
            secrect_key="sk_test",
            disable_batch=True,
        )

        mock_simple.assert_called_once()


class TestOtelEnvironmentVariables:
    """Tests for environment variable handling in OTel modules."""

    @patch("tmam.otel.tracing.trace")
    @patch("tmam.otel.tracing.TracerProvider")
    @patch("tmam.otel.tracing.OTLPSpanExporter")
    @patch("tmam.otel.tracing.BatchSpanProcessor")
    def test_tracing_sets_endpoint_env_var(
        self, mock_batch, mock_exporter, mock_provider, mock_trace
    ):
        """Test that OTEL_EXPORTER_OTLP_ENDPOINT is set."""
        from tmam.otel import tracing

        tracing.TRACER_SET = False

        mock_tracer = MagicMock()
        mock_trace.get_tracer_provider.return_value.get_tracer.return_value = (
            mock_tracer
        )
        mock_trace.get_tracer_provider.return_value.add_span_processor = MagicMock()

        tracing.setup_tracing(
            application_name="test-app",
            environment="test",
            tracer=None,
            url="https://custom.endpoint.com",
            public_key="pk",
            secrect_key="sk",
            disable_batch=False,
        )

        assert (
            os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
            == "https://custom.endpoint.com"
        )

    @patch("tmam.otel.tracing.trace")
    @patch("tmam.otel.tracing.TracerProvider")
    @patch("tmam.otel.tracing.OTLPSpanExporter")
    @patch("tmam.otel.tracing.BatchSpanProcessor")
    def test_tracing_sets_headers_env_var(
        self, mock_batch, mock_exporter, mock_provider, mock_trace
    ):
        """Test that OTEL_EXPORTER_OTLP_HEADERS is set with keys."""
        from tmam.otel import tracing

        tracing.TRACER_SET = False

        mock_tracer = MagicMock()
        mock_trace.get_tracer_provider.return_value.get_tracer.return_value = (
            mock_tracer
        )
        mock_trace.get_tracer_provider.return_value.add_span_processor = MagicMock()

        tracing.setup_tracing(
            application_name="test-app",
            environment="test",
            tracer=None,
            url="https://api.test.com",
            public_key="test_public_key",
            secrect_key="test_secret_key",
            disable_batch=False,
        )

        headers = os.environ.get("OTEL_EXPORTER_OTLP_HEADERS")
        assert headers is not None
        assert "X-Public-Key=test_public_key" in headers
        assert "X-Secret-Key=test_secret_key" in headers


class TestMetricBuckets:
    """Tests for metric bucket configurations."""

    def test_operation_duration_buckets(self):
        """Test that operation duration buckets are properly defined."""
        from tmam.otel.metrics import _GEN_AI_CLIENT_OPERATION_DURATION_BUCKETS

        assert len(_GEN_AI_CLIENT_OPERATION_DURATION_BUCKETS) > 0
        # Buckets should be in ascending order
        for i in range(1, len(_GEN_AI_CLIENT_OPERATION_DURATION_BUCKETS)):
            assert (
                _GEN_AI_CLIENT_OPERATION_DURATION_BUCKETS[i]
                > _GEN_AI_CLIENT_OPERATION_DURATION_BUCKETS[i - 1]
            )

    def test_token_usage_buckets(self):
        """Test that token usage buckets are properly defined."""
        from tmam.otel.metrics import _GEN_AI_CLIENT_TOKEN_USAGE_BUCKETS

        assert len(_GEN_AI_CLIENT_TOKEN_USAGE_BUCKETS) > 0
        # Buckets should be in ascending order
        for i in range(1, len(_GEN_AI_CLIENT_TOKEN_USAGE_BUCKETS)):
            assert (
                _GEN_AI_CLIENT_TOKEN_USAGE_BUCKETS[i]
                > _GEN_AI_CLIENT_TOKEN_USAGE_BUCKETS[i - 1]
            )

    def test_tbt_buckets(self):
        """Test that TBT buckets are properly defined."""
        from tmam.otel.metrics import _GEN_AI_SERVER_TBT

        assert len(_GEN_AI_SERVER_TBT) > 0
        # Buckets should be in ascending order
        for i in range(1, len(_GEN_AI_SERVER_TBT)):
            assert _GEN_AI_SERVER_TBT[i] > _GEN_AI_SERVER_TBT[i - 1]

    def test_ttft_buckets(self):
        """Test that TTFT buckets are properly defined."""
        from tmam.otel.metrics import _GEN_AI_SERVER_TFTT

        assert len(_GEN_AI_SERVER_TFTT) > 0
        # Buckets should be in ascending order
        for i in range(1, len(_GEN_AI_SERVER_TFTT)):
            assert _GEN_AI_SERVER_TFTT[i] > _GEN_AI_SERVER_TFTT[i - 1]

    def test_db_duration_buckets(self):
        """Test that DB duration buckets are properly defined."""
        from tmam.otel.metrics import _DB_CLIENT_OPERATION_DURATION_BUCKETS

        assert len(_DB_CLIENT_OPERATION_DURATION_BUCKETS) > 0
        # Buckets should be in ascending order
        for i in range(1, len(_DB_CLIENT_OPERATION_DURATION_BUCKETS)):
            assert (
                _DB_CLIENT_OPERATION_DURATION_BUCKETS[i]
                > _DB_CLIENT_OPERATION_DURATION_BUCKETS[i - 1]
            )
