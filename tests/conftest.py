"""
Pytest configuration and shared fixtures for TMAM SDK tests.
"""

import os
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture(autouse=True)
def reset_env_vars():
    """Reset environment variables before each test."""
    env_vars_to_clear = [
        "TMAM_URL",
        "TMAM_PUBLIC_KEY",
        "TMAM_SECRET_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_EXPORTER_OTLP_HEADERS",
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT",
        "HAYSTACK_AUTO_TRACE_ENABLED",
    ]
    original_values = {}
    for var in env_vars_to_clear:
        original_values[var] = os.environ.get(var)
        if var in os.environ:
            del os.environ[var]

    yield

    # Restore original values
    for var, value in original_values.items():
        if value is not None:
            os.environ[var] = value
        elif var in os.environ:
            del os.environ[var]


@pytest.fixture
def mock_tracer():
    """Create a mock tracer for testing."""
    tracer = MagicMock()
    span = MagicMock()
    span.__enter__ = MagicMock(return_value=span)
    span.__exit__ = MagicMock(return_value=False)
    tracer.start_as_current_span = MagicMock(return_value=span)
    return tracer


@pytest.fixture
def mock_span():
    """Create a mock span for testing."""
    span = MagicMock()
    span.set_attribute = MagicMock()
    span.set_attributes = MagicMock()
    span.set_status = MagicMock()
    span.record_exception = MagicMock()
    span.add_event = MagicMock()
    span.end = MagicMock()
    return span


@pytest.fixture
def mock_meter():
    """Create a mock meter for testing."""
    meter = MagicMock()
    counter = MagicMock()
    histogram = MagicMock()
    meter.create_counter = MagicMock(return_value=counter)
    meter.create_histogram = MagicMock(return_value=histogram)
    return meter


@pytest.fixture
def mock_event_logger():
    """Create a mock event logger for testing."""
    return MagicMock()


@pytest.fixture
def sample_pricing_info():
    """Return sample pricing information for testing."""
    return {
        "chat": {
            "gpt-4o": {"promptPrice": 0.005, "completionPrice": 0.015},
            "gpt-4o-mini": {"promptPrice": 0.00015, "completionPrice": 0.0006},
            "claude-3-opus-20240229": {"promptPrice": 0.015, "completionPrice": 0.075},
        },
        "embeddings": {
            "text-embedding-ada-002": 0.0001,
            "text-embedding-3-small": 0.00002,
        },
        "images": {
            "dall-e-3": {
                "standard": {"1024x1024": 0.04, "1024x1792": 0.08},
                "hd": {"1024x1024": 0.08, "1024x1792": 0.12},
            }
        },
        "audio": {
            "tts-1": 0.015,
            "tts-1-hd": 0.03,
            "whisper-1": 0.006,
        },
    }


@pytest.fixture
def mock_openai_response():
    """Create a mock OpenAI chat completion response."""
    response = MagicMock()
    response.id = "chatcmpl-test123"
    response.model = "gpt-4o"
    response.choices = [MagicMock()]
    response.choices[0].message.content = "This is a test response."
    response.choices[0].finish_reason = "stop"
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = 20
    response.usage.total_tokens = 30
    return response


@pytest.fixture
def mock_anthropic_response():
    """Create a mock Anthropic messages response."""
    response = MagicMock()
    response.id = "msg_test123"
    response.model = "claude-3-opus-20240229"
    response.content = [MagicMock()]
    response.content[0].type = "text"
    response.content[0].text = "This is a test response from Claude."
    response.stop_reason = "end_turn"
    response.usage.input_tokens = 15
    response.usage.output_tokens = 25
    return response


@pytest.fixture
def sample_messages():
    """Return sample messages for chat completion tests."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
    ]


@pytest.fixture
def sample_contexts():
    """Return sample contexts for evaluation tests."""
    return [
        "The capital of France is Paris.",
        "Einstein won the Nobel Prize for Physics in 1921.",
        "Water boils at 100 degrees Celsius at sea level.",
    ]


@pytest.fixture
def tmam_config_env():
    """Set up environment variables for TMAM configuration."""
    os.environ["TMAM_URL"] = "https://api.tmam.test/v1"
    os.environ["TMAM_PUBLIC_KEY"] = "test-public-key"
    os.environ["TMAM_SECRET_KEY"] = "test-secret-key"
    yield
    del os.environ["TMAM_URL"]
    del os.environ["TMAM_PUBLIC_KEY"]
    del os.environ["TMAM_SECRET_KEY"]


@pytest.fixture
def mock_requests_post():
    """Mock requests.post for API calls."""
    with patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": {"result": "success"}}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response
        yield mock_post


@pytest.fixture
def mock_requests_get():
    """Mock requests.get for API calls."""
    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        yield mock_get
