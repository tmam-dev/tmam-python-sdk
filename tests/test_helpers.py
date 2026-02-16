"""
Tests for the tmam.__helpers module.
"""

import pytest
import os
import json
from unittest.mock import MagicMock, patch, mock_open

import sys

sys.path.insert(0, "/home/claude/sdk/sdk/src")

from tmam.__helpers import (
    response_as_dict,
    get_env_variable,
    general_tokens,
    get_chat_model_cost,
    get_embed_model_cost,
    get_image_model_cost,
    get_audio_model_cost,
    fetch_pricing_info,
    handle_exception,
    calculate_ttft,
    calculate_tbt,
    create_metrics_attributes,
    set_server_address_and_port,
    extract_and_format_input,
    concatenate_all_contents,
)


class TestResponseAsDict:
    """Tests for response_as_dict function."""

    def test_dict_input_returns_same_dict(self):
        """Test that dict input returns the same dict."""
        data = {"key": "value", "number": 42}
        result = response_as_dict(data)
        assert result == data

    def test_object_with_model_dump(self):
        """Test object with model_dump method (Pydantic models)."""
        mock_obj = MagicMock()
        mock_obj.model_dump.return_value = {"field": "value"}
        result = response_as_dict(mock_obj)
        assert result == {"field": "value"}
        mock_obj.model_dump.assert_called_once()

    def test_object_with_parse_method(self):
        """Test object with parse method."""
        inner_obj = MagicMock()
        inner_obj.model_dump.return_value = {"parsed": "data"}

        mock_obj = MagicMock(spec=["parse"])
        mock_obj.parse.return_value = inner_obj

        result = response_as_dict(mock_obj)
        assert result == {"parsed": "data"}

    def test_plain_object_returns_itself(self):
        """Test that plain object without special methods returns itself."""

        class PlainObject:
            pass

        obj = PlainObject()
        result = response_as_dict(obj)
        assert result is obj


class TestGetEnvVariable:
    """Tests for get_env_variable function."""

    def test_arg_value_takes_precedence(self):
        """Test that argument value takes precedence over env variable."""
        os.environ["TEST_VAR"] = "env_value"
        result = get_env_variable("TEST_VAR", "arg_value", "error")
        assert result == "arg_value"
        del os.environ["TEST_VAR"]

    def test_env_variable_used_when_arg_none(self):
        """Test that env variable is used when arg is None."""
        os.environ["TEST_VAR"] = "env_value"
        result = get_env_variable("TEST_VAR", None, "error")
        assert result == "env_value"
        del os.environ["TEST_VAR"]

    def test_raises_error_when_both_missing(self):
        """Test that RuntimeError is raised when both arg and env are missing."""
        with pytest.raises(RuntimeError) as exc_info:
            get_env_variable("NONEXISTENT_VAR", None, "Missing variable!")
        assert "Missing variable!" in str(exc_info.value)


class TestGeneralTokens:
    """Tests for general_tokens function."""

    def test_empty_string(self):
        """Test token count for empty string."""
        assert general_tokens("") == 0

    def test_short_string(self):
        """Test token count for short string."""
        assert general_tokens("Hi") == 1

    def test_longer_string(self):
        """Test token count for longer string."""
        # 20 characters should be ~10 tokens
        text = "Hello, world! Test."
        result = general_tokens(text)
        assert result == 10  # ceil(19/2) = 10


class TestGetChatModelCost:
    """Tests for get_chat_model_cost function."""

    def test_valid_model_cost_calculation(self, sample_pricing_info):
        """Test cost calculation for valid model."""
        cost = get_chat_model_cost("gpt-4o", sample_pricing_info, 1000, 500)
        expected = (1000 / 1000 * 0.005) + (500 / 1000 * 0.015)
        assert cost == expected

    def test_unknown_model_returns_zero(self, sample_pricing_info):
        """Test that unknown model returns 0 cost."""
        cost = get_chat_model_cost("unknown-model", sample_pricing_info, 1000, 500)
        assert cost == 0

    def test_empty_pricing_info(self):
        """Test with empty pricing info."""
        cost = get_chat_model_cost("gpt-4o", {}, 1000, 500)
        assert cost == 0


class TestGetEmbedModelCost:
    """Tests for get_embed_model_cost function."""

    def test_valid_embedding_cost(self, sample_pricing_info):
        """Test cost calculation for valid embedding model."""
        cost = get_embed_model_cost("text-embedding-ada-002", sample_pricing_info, 1000)
        expected = (1000 / 1000) * 0.0001
        assert cost == expected

    def test_unknown_model_returns_zero(self, sample_pricing_info):
        """Test that unknown model returns 0 cost."""
        cost = get_embed_model_cost("unknown-embed", sample_pricing_info, 1000)
        assert cost == 0


class TestGetImageModelCost:
    """Tests for get_image_model_cost function."""

    def test_valid_image_cost(self, sample_pricing_info):
        """Test cost calculation for valid image model."""
        cost = get_image_model_cost(
            "dall-e-3", sample_pricing_info, "1024x1024", "standard"
        )
        assert cost == 0.04

    def test_hd_quality_cost(self, sample_pricing_info):
        """Test HD quality cost."""
        cost = get_image_model_cost("dall-e-3", sample_pricing_info, "1024x1024", "hd")
        assert cost == 0.08

    def test_unknown_model_returns_zero(self, sample_pricing_info):
        """Test that unknown model returns 0 cost."""
        cost = get_image_model_cost(
            "unknown", sample_pricing_info, "1024x1024", "standard"
        )
        assert cost == 0


class TestGetAudioModelCost:
    """Tests for get_audio_model_cost function."""

    def test_tts_cost_with_prompt(self, sample_pricing_info):
        """Test TTS cost calculation with prompt."""
        prompt = "Hello, this is a test." * 100  # 2300 characters
        cost = get_audio_model_cost("tts-1", sample_pricing_info, prompt)
        expected = (len(prompt) / 1000) * 0.015
        assert cost == expected

    def test_transcription_cost_with_duration(self, sample_pricing_info):
        """Test transcription cost calculation with duration."""
        cost = get_audio_model_cost("whisper-1", sample_pricing_info, None, duration=60)
        expected = 60 * 0.006
        assert cost == expected

    def test_unknown_model_returns_zero(self, sample_pricing_info):
        """Test that unknown model returns 0 cost."""
        cost = get_audio_model_cost("unknown-audio", sample_pricing_info, "test")
        assert cost == 0


class TestFetchPricingInfo:
    """Tests for fetch_pricing_info function."""

    def test_fetch_from_file(self, tmp_path):
        """Test fetching pricing info from file."""
        pricing_data = {"chat": {"model": {"promptPrice": 0.01}}}
        pricing_file = tmp_path / "pricing.json"
        pricing_file.write_text(json.dumps(pricing_data))

        result = fetch_pricing_info(str(pricing_file))
        assert result == pricing_data

    def test_fetch_from_nonexistent_file(self):
        """Test fetching from nonexistent file returns empty dict."""
        result = fetch_pricing_info("/nonexistent/path/pricing.json")
        assert result == {}

    @patch("requests.get")
    def test_fetch_from_url(self, mock_get):
        """Test fetching pricing info from URL."""
        pricing_data = {"chat": {"model": {"promptPrice": 0.01}}}
        mock_response = MagicMock()
        mock_response.json.return_value = pricing_data
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = fetch_pricing_info("https://example.com/pricing.json")
        assert result == pricing_data

    @patch("requests.get")
    def test_fetch_from_default_url(self, mock_get):
        """Test fetching from default URL when no argument provided."""
        pricing_data = {"chat": {}}
        mock_response = MagicMock()
        mock_response.json.return_value = pricing_data
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = fetch_pricing_info(None)
        assert result == pricing_data
        mock_get.assert_called_once()


class TestHandleException:
    """Tests for handle_exception function."""

    def test_records_exception_and_sets_error_status(self, mock_span):
        """Test that exception is recorded and status is set to error."""
        exc = ValueError("Test error")
        handle_exception(mock_span, exc)

        mock_span.record_exception.assert_called_once_with(exc)
        mock_span.set_status.assert_called_once()


class TestCalculateTtft:
    """Tests for calculate_ttft function."""

    def test_with_timestamps(self):
        """Test TTFT calculation with timestamps."""
        timestamps = [1.5, 2.0, 2.5]
        start_time = 1.0
        result = calculate_ttft(timestamps, start_time)
        assert result == 0.5

    def test_empty_timestamps(self):
        """Test TTFT with empty timestamps returns 0."""
        result = calculate_ttft([], 1.0)
        assert result == 0.0


class TestCalculateTbt:
    """Tests for calculate_tbt function."""

    def test_multiple_timestamps(self):
        """Test TBT calculation with multiple timestamps."""
        timestamps = [1.0, 1.1, 1.2, 1.3]
        result = calculate_tbt(timestamps)
        assert result == pytest.approx(0.1, abs=0.001)

    def test_single_timestamp(self):
        """Test TBT with single timestamp returns 0."""
        result = calculate_tbt([1.0])
        assert result == 0.0

    def test_empty_timestamps(self):
        """Test TBT with empty timestamps returns 0."""
        result = calculate_tbt([])
        assert result == 0.0


class TestCreateMetricsAttributes:
    """Tests for create_metrics_attributes function."""

    def test_creates_correct_attributes(self):
        """Test that all expected attributes are created."""
        attrs = create_metrics_attributes(
            service_name="test-service",
            deployment_environment="test",
            operation="chat",
            system="openai",
            request_model="gpt-4o",
            server_address="api.openai.com",
            server_port=443,
            response_model="gpt-4o",
        )

        assert attrs["service.name"] == "test-service"
        assert attrs["deployment.environment"] == "test"
        assert attrs["gen_ai.operation.name"] == "chat"
        assert attrs["gen_ai.system"] == "openai"
        assert attrs["gen_ai.request.model"] == "gpt-4o"
        assert attrs["server.address"] == "api.openai.com"
        assert attrs["server.port"] == 443


class TestSetServerAddressAndPort:
    """Tests for set_server_address_and_port function."""

    def test_with_base_url_string(self):
        """Test extraction from string base_url."""
        mock_client = MagicMock()
        mock_client._client.base_url = "https://api.example.com:8080/v1"

        address, port = set_server_address_and_port(mock_client, "default.com", 443)
        assert address == "api.example.com"
        assert port == 8080

    def test_with_default_port(self):
        """Test default port when not specified in URL."""
        mock_client = MagicMock()
        mock_client._client.base_url = "https://api.example.com/v1"

        address, port = set_server_address_and_port(mock_client, "default.com", 443)
        assert address == "api.example.com"
        assert port == 443

    def test_with_no_base_url(self):
        """Test defaults are used when no base_url."""
        mock_client = MagicMock(spec=[])

        address, port = set_server_address_and_port(mock_client, "default.com", 443)
        assert address == "default.com"
        assert port == 443


class TestExtractAndFormatInput:
    """Tests for extract_and_format_input function."""

    def test_extract_user_message(self):
        """Test extraction of user message."""
        messages = [{"role": "user", "content": "Hello"}]
        result = extract_and_format_input(messages)
        assert result["user"]["role"] == "user"
        assert result["user"]["content"] == "Hello"

    def test_extract_system_message(self):
        """Test extraction of system message."""
        messages = [{"role": "system", "content": "You are helpful"}]
        result = extract_and_format_input(messages)
        assert result["system"]["role"] == "system"
        assert result["system"]["content"] == "You are helpful"

    def test_extract_multiple_messages(self):
        """Test extraction of multiple messages."""
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        result = extract_and_format_input(messages)
        assert result["system"]["content"] == "You are helpful"
        assert result["user"]["content"] == "Hi"
        assert result["assistant"]["content"] == "Hello!"

    def test_concatenate_same_role(self):
        """Test that multiple messages with same role are concatenated."""
        messages = [
            {"role": "user", "content": "First"},
            {"role": "user", "content": "Second"},
        ]
        result = extract_and_format_input(messages)
        assert "First" in result["user"]["content"]
        assert "Second" in result["user"]["content"]

    def test_list_content(self):
        """Test handling of list content."""
        messages = [{"role": "user", "content": ["Part 1", "Part 2"]}]
        result = extract_and_format_input(messages)
        assert "Part 1" in result["user"]["content"]
        assert "Part 2" in result["user"]["content"]


class TestConcatenateAllContents:
    """Tests for concatenate_all_contents function."""

    def test_concatenate_contents(self):
        """Test concatenation of all content fields."""
        formatted = {
            "user": {"role": "user", "content": "Hello"},
            "assistant": {"role": "assistant", "content": "Hi there"},
            "system": {"role": "system", "content": ""},
        }
        result = concatenate_all_contents(formatted)
        assert "Hello" in result
        assert "Hi there" in result

    def test_empty_contents_excluded(self):
        """Test that empty contents are excluded."""
        formatted = {
            "user": {"role": "user", "content": "Hello"},
            "system": {"role": "system", "content": ""},
        }
        result = concatenate_all_contents(formatted)
        assert result.strip() == "Hello"
