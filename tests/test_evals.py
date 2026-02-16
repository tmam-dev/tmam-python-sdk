"""
Tests for the tmam.evals module (hallucination, toxicity, bias detection).
"""

import pytest
import json
from unittest.mock import MagicMock, patch

import sys

sys.path.insert(0, "/home/claude/sdk/sdk/src")


class TestEvalsUtils:
    """Tests for evals utility functions."""

    def test_json_output_model(self):
        """Test JsonOutput Pydantic model."""
        from tmam.evals.utils import JsonOutput

        output = JsonOutput(
            verdict="yes",
            evaluation="hallucination",
            score=0.8,
            classification="factual_inaccuracy",
            explanation="The text contains incorrect facts.",
        )

        assert output.verdict == "yes"
        assert output.evaluation == "hallucination"
        assert output.score == 0.8
        assert output.classification == "factual_inaccuracy"
        assert output.explanation == "The text contains incorrect facts."

    def test_setup_provider_openai(self):
        """Test setup_provider for OpenAI."""
        import os
        from tmam.evals.utils import setup_provider

        os.environ["OPENAI_API_KEY"] = "test-key"

        api_key, model, base_url = setup_provider(
            provider="openai", api_key=None, model="gpt-4o", base_url=None
        )

        assert api_key == "test-key"
        assert model == "gpt-4o"

        del os.environ["OPENAI_API_KEY"]

    def test_setup_provider_anthropic(self):
        """Test setup_provider for Anthropic."""
        import os
        from tmam.evals.utils import setup_provider

        os.environ["ANTHROPIC_API_KEY"] = "anthropic-key"

        api_key, model, base_url = setup_provider(
            provider="anthropic",
            api_key=None,
            model="claude-3-opus-20240229",
            base_url=None,
        )

        assert api_key == "anthropic-key"

        del os.environ["ANTHROPIC_API_KEY"]

    def test_setup_provider_unsupported(self):
        """Test setup_provider raises for unsupported provider."""
        from tmam.evals.utils import setup_provider

        with pytest.raises(ValueError) as exc_info:
            setup_provider(
                provider="unsupported_provider",
                api_key="key",
                model="model",
                base_url=None,
            )

        assert "Unsupported provider" in str(exc_info.value)

    def test_setup_provider_missing_api_key(self):
        """Test setup_provider raises when API key is missing."""
        import os
        from tmam.evals.utils import setup_provider

        # Ensure environment variable is not set
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]

        with pytest.raises(ValueError) as exc_info:
            setup_provider(provider="openai", api_key=None, model=None, base_url=None)

        assert "API key required" in str(exc_info.value)

    def test_setup_provider_none(self):
        """Test setup_provider returns None for all when provider is None."""
        from tmam.evals.utils import setup_provider

        api_key, model, base_url = setup_provider(
            provider=None, api_key=None, model=None, base_url=None
        )

        assert api_key is None
        assert model is None
        assert base_url is None

    def test_format_prompt(self):
        """Test format_prompt function."""
        from tmam.evals.utils import format_prompt

        system_prompt = """
        Contexts: {{context}}
        Text: {{text}}
        Prompt: {{prompt}}
        """

        result = format_prompt(
            system_prompt=system_prompt,
            prompt="What is the capital?",
            contexts=["Paris is the capital of France."],
            text="The capital is Lyon.",
        )

        assert "Paris is the capital of France" in result
        assert "The capital is Lyon" in result
        assert "What is the capital?" in result

    def test_parse_llm_response_string(self):
        """Test parse_llm_response with JSON string."""
        from tmam.evals.utils import parse_llm_response, JsonOutput

        response = json.dumps(
            {
                "verdict": "yes",
                "evaluation": "hallucination",
                "score": 0.9,
                "classification": "factual_inaccuracy",
                "explanation": "Incorrect information detected.",
            }
        )

        result = parse_llm_response(response)

        assert isinstance(result, JsonOutput)
        assert result.verdict == "yes"
        assert result.score == 0.9

    def test_parse_llm_response_dict(self):
        """Test parse_llm_response with dict."""
        from tmam.evals.utils import parse_llm_response, JsonOutput

        response = {
            "verdict": "no",
            "evaluation": "toxicity_detection",
            "score": 0.1,
            "classification": "none",
            "explanation": "No issues detected.",
        }

        result = parse_llm_response(response)

        assert isinstance(result, JsonOutput)
        assert result.verdict == "no"
        assert result.score == 0.1

    def test_parse_llm_response_invalid(self):
        """Test parse_llm_response with invalid input."""
        from tmam.evals.utils import parse_llm_response, JsonOutput

        result = parse_llm_response("invalid json {{{")

        assert isinstance(result, JsonOutput)
        assert result.score == 0
        assert result.verdict == "no"


class TestHallucinationDetection:
    """Tests for Hallucination detection class."""

    def test_hallucination_init(self):
        """Test Hallucination class initialization."""
        import os
        from tmam.evals.hallucination import Hallucination

        os.environ["OPENAI_API_KEY"] = "test-key"

        detector = Hallucination(
            provider="openai", model="gpt-4o-mini", threshold_score=0.6
        )

        assert detector.provider == "openai"
        assert detector.model == "gpt-4o-mini"
        assert detector.threshold_score == 0.6

        del os.environ["OPENAI_API_KEY"]

    def test_hallucination_init_no_provider(self):
        """Test Hallucination raises without provider."""
        from tmam.evals.hallucination import Hallucination

        with pytest.raises(ValueError) as exc_info:
            Hallucination(provider=None)

        assert "LLM provider must be specified" in str(exc_info.value)

    def test_hallucination_system_prompt_includes_categories(self):
        """Test that system prompt includes all hallucination categories."""
        from tmam.evals.hallucination import get_system_prompt

        prompt = get_system_prompt()

        assert "factual_inaccuracy" in prompt
        assert "nonsensical_response" in prompt
        assert "gibberish" in prompt
        assert "contradiction" in prompt

    def test_hallucination_system_prompt_custom_categories(self):
        """Test that custom categories are included in system prompt."""
        from tmam.evals.hallucination import get_system_prompt

        custom_categories = {"custom_category": "A custom hallucination type"}

        prompt = get_system_prompt(custom_categories=custom_categories)

        assert "custom_category" in prompt
        assert "A custom hallucination type" in prompt

    @patch("tmam.evals.hallucination.llm_response")
    @patch("tmam.evals.hallucination.setup_provider")
    def test_hallucination_measure(self, mock_setup, mock_llm):
        """Test Hallucination.measure method."""
        from tmam.evals.hallucination import Hallucination

        mock_setup.return_value = ("api-key", "gpt-4o-mini", None)
        mock_llm.return_value = json.dumps(
            {
                "verdict": "yes",
                "evaluation": "hallucination",
                "score": 0.8,
                "classification": "factual_inaccuracy",
                "explanation": "Date is incorrect.",
            }
        )

        detector = Hallucination(provider="openai")
        result = detector.measure(
            prompt="When was Einstein born?",
            contexts=["Einstein was born in 1879."],
            text="Einstein was born in 1880.",
        )

        assert result.verdict == "yes"
        assert result.score == 0.8
        assert result.classification == "factual_inaccuracy"


class TestToxicityDetection:
    """Tests for Toxicity detection class."""

    def test_toxicity_init(self):
        """Test ToxicityDetector class initialization."""
        import os
        from tmam.evals.toxicity import ToxicityDetector

        os.environ["OPENAI_API_KEY"] = "test-key"

        detector = ToxicityDetector(
            provider="openai", model="gpt-4o-mini", threshold_score=0.5
        )

        assert detector.provider == "openai"
        assert detector.threshold_score == 0.5

        del os.environ["OPENAI_API_KEY"]

    def test_toxicity_init_no_provider(self):
        """Test ToxicityDetector raises without provider."""
        from tmam.evals.toxicity import ToxicityDetector

        with pytest.raises(ValueError) as exc_info:
            ToxicityDetector(provider=None)

        assert "LLM provider must be specified" in str(exc_info.value)

    def test_toxicity_system_prompt_includes_categories(self):
        """Test that system prompt includes all toxicity categories."""
        from tmam.evals.toxicity import get_system_prompt

        prompt = get_system_prompt()

        assert "threat" in prompt
        assert "dismissive" in prompt
        assert "hate" in prompt
        assert "mockery" in prompt
        assert "personal_attack" in prompt

    def test_toxicity_system_prompt_custom_categories(self):
        """Test that custom categories are included in system prompt."""
        from tmam.evals.toxicity import get_system_prompt

        custom_categories = {"profanity": "Use of explicit language"}

        prompt = get_system_prompt(custom_categories=custom_categories)

        assert "profanity" in prompt
        assert "Use of explicit language" in prompt

    @patch("tmam.evals.toxicity.llm_response")
    @patch("tmam.evals.toxicity.setup_provider")
    def test_toxicity_measure(self, mock_setup, mock_llm):
        """Test ToxicityDetector.measure method."""
        from tmam.evals.toxicity import ToxicityDetector

        mock_setup.return_value = ("api-key", "gpt-4o-mini", None)
        mock_llm.return_value = json.dumps(
            {
                "verdict": "yes",
                "evaluation": "toxicity_detection",
                "score": 0.9,
                "classification": "personal_attack",
                "explanation": "The text contains a personal attack.",
            }
        )

        detector = ToxicityDetector(provider="openai")
        result = detector.measure(
            prompt="Provide feedback",
            contexts=[],
            text="You're an idiot who doesn't know anything!",
        )

        assert result.verdict == "yes"
        assert result.score == 0.9
        assert result.classification == "personal_attack"


class TestBiasDetection:
    """Tests for Bias detection class."""

    def test_bias_init(self):
        """Test BiasDetector class initialization."""
        import os
        from tmam.evals.bias_detection import BiasDetector

        os.environ["OPENAI_API_KEY"] = "test-key"

        detector = BiasDetector(
            provider="openai", model="gpt-4o-mini", threshold_score=0.5
        )

        assert detector.provider == "openai"
        assert detector.threshold_score == 0.5

        del os.environ["OPENAI_API_KEY"]

    def test_bias_init_no_provider(self):
        """Test BiasDetector raises without provider."""
        from tmam.evals.bias_detection import BiasDetector

        with pytest.raises(ValueError) as exc_info:
            BiasDetector(provider=None)

        assert "LLM provider must be specified" in str(exc_info.value)

    def test_bias_system_prompt_includes_categories(self):
        """Test that system prompt includes all bias categories."""
        from tmam.evals.bias_detection import get_system_prompt

        prompt = get_system_prompt()

        assert "sexual_orientation" in prompt
        assert "age" in prompt
        assert "disability" in prompt
        assert "physical_appearance" in prompt
        assert "religion" in prompt
        assert "pregnancy_status" in prompt
        assert "marital_status" in prompt
        assert "nationality" in prompt
        assert "gender" in prompt
        assert "ethnicity" in prompt
        assert "socioeconomic_status" in prompt

    def test_bias_system_prompt_custom_categories(self):
        """Test that custom categories are included in system prompt."""
        from tmam.evals.bias_detection import get_system_prompt

        custom_categories = {"political_affiliation": "Bias based on political beliefs"}

        prompt = get_system_prompt(custom_categories=custom_categories)

        assert "political_affiliation" in prompt
        assert "Bias based on political beliefs" in prompt

    @patch("tmam.evals.bias_detection.llm_response")
    @patch("tmam.evals.bias_detection.setup_provider")
    def test_bias_measure(self, mock_setup, mock_llm):
        """Test BiasDetector.measure method."""
        from tmam.evals.bias_detection import BiasDetector

        mock_setup.return_value = ("api-key", "gpt-4o-mini", None)
        mock_llm.return_value = json.dumps(
            {
                "verdict": "yes",
                "evaluation": "bias_detection",
                "score": 0.7,
                "classification": "age",
                "explanation": "The text contains age-related bias.",
            }
        )

        detector = BiasDetector(provider="openai")
        result = detector.measure(
            prompt="Describe workers",
            contexts=["Workers of all ages contribute."],
            text="Older workers are less productive.",
        )

        assert result.verdict == "yes"
        assert result.score == 0.7
        assert result.classification == "age"


class TestAllEvals:
    """Tests for All evaluations class."""

    def test_all_init(self):
        """Test All class initialization."""
        import os
        from tmam.evals.all import All

        os.environ["OPENAI_API_KEY"] = "test-key"

        detector = All(provider="openai", model="gpt-4o-mini", threshold_score=0.5)

        assert detector.provider == "openai"
        assert detector.threshold_score == 0.5

        del os.environ["OPENAI_API_KEY"]

    def test_all_init_no_provider(self):
        """Test All raises without provider."""
        from tmam.evals.all import All

        with pytest.raises(ValueError) as exc_info:
            All(provider=None)

        assert "LLM provider must be specified" in str(exc_info.value)

    def test_all_system_prompt_includes_all_categories(self):
        """Test that system prompt includes categories from all eval types."""
        from tmam.evals.all import get_system_prompt

        prompt = get_system_prompt()

        # Bias categories
        assert "sexual_orientation" in prompt
        assert "gender" in prompt

        # Toxicity categories
        assert "threat" in prompt
        assert "hate" in prompt

        # Hallucination categories
        assert "factual_inaccuracy" in prompt
        assert "contradiction" in prompt

    @patch("tmam.evals.all.llm_response")
    @patch("tmam.evals.all.setup_provider")
    def test_all_measure(self, mock_setup, mock_llm):
        """Test All.measure method."""
        from tmam.evals.all import All

        mock_setup.return_value = ("api-key", "gpt-4o-mini", None)
        mock_llm.return_value = json.dumps(
            {
                "verdict": "yes",
                "evaluation": "hallucination",
                "score": 0.8,
                "classification": "factual_inaccuracy",
                "explanation": "The text contains incorrect facts.",
            }
        )

        detector = All(provider="openai")
        result = detector.measure(
            prompt="Describe the event",
            contexts=["The event occurred in 2020."],
            text="The event occurred in 2025.",
        )

        assert result.verdict == "yes"
        assert result.score == 0.8
        assert result.evaluation == "hallucination"


class TestEvalsMetrics:
    """Tests for evaluation metrics collection."""

    @patch("tmam.evals.utils.get_meter")
    def test_eval_metrics_creates_counter(self, mock_get_meter):
        """Test that eval_metrics creates a counter."""
        from tmam.evals.utils import eval_metrics

        mock_meter = MagicMock()
        mock_counter = MagicMock()
        mock_meter.create_counter.return_value = mock_counter
        mock_get_meter.return_value = mock_meter

        result = eval_metrics()

        mock_meter.create_counter.assert_called_once()
        assert result is mock_counter

    def test_eval_metric_attributes(self):
        """Test eval_metric_attributes returns correct structure."""
        from tmam.evals.utils import eval_metric_attributes

        attrs = eval_metric_attributes(
            verdict="yes",
            score=0.8,
            validator="hallucination",
            classification="factual_inaccuracy",
            explanation="Test explanation",
        )

        assert attrs["evals.verdict"] == "yes"
        assert attrs["evals.score"] == 0.8
        assert attrs["evals.validator"] == "hallucination"
        assert attrs["evals.classification"] == "factual_inaccuracy"
        assert attrs["evals.explanation"] == "Test explanation"


class TestThresholdScoring:
    """Tests for threshold-based verdict determination."""

    @patch("tmam.evals.hallucination.llm_response")
    @patch("tmam.evals.hallucination.setup_provider")
    def test_verdict_yes_above_threshold(self, mock_setup, mock_llm):
        """Test that verdict is 'yes' when score is above threshold."""
        from tmam.evals.hallucination import Hallucination

        mock_setup.return_value = ("api-key", "gpt-4o-mini", None)
        mock_llm.return_value = json.dumps(
            {
                "verdict": "no",  # LLM might return different verdict
                "evaluation": "hallucination",
                "score": 0.8,  # Above default threshold of 0.5
                "classification": "factual_inaccuracy",
                "explanation": "Test",
            }
        )

        detector = Hallucination(provider="openai", threshold_score=0.5)
        result = detector.measure(prompt="Test", contexts=["Context"], text="Text")

        # Our code overrides the verdict based on threshold
        assert result.verdict == "yes"

    @patch("tmam.evals.hallucination.llm_response")
    @patch("tmam.evals.hallucination.setup_provider")
    def test_verdict_no_below_threshold(self, mock_setup, mock_llm):
        """Test that verdict is 'no' when score is below threshold."""
        from tmam.evals.hallucination import Hallucination

        mock_setup.return_value = ("api-key", "gpt-4o-mini", None)
        mock_llm.return_value = json.dumps(
            {
                "verdict": "yes",  # LLM might return different verdict
                "evaluation": "hallucination",
                "score": 0.3,  # Below default threshold of 0.5
                "classification": "none",
                "explanation": "Test",
            }
        )

        detector = Hallucination(provider="openai", threshold_score=0.5)
        result = detector.measure(prompt="Test", contexts=["Context"], text="Text")

        # Our code overrides the verdict based on threshold
        assert result.verdict == "no"

    @patch("tmam.evals.hallucination.llm_response")
    @patch("tmam.evals.hallucination.setup_provider")
    def test_custom_threshold(self, mock_setup, mock_llm):
        """Test custom threshold values."""
        from tmam.evals.hallucination import Hallucination

        mock_setup.return_value = ("api-key", "gpt-4o-mini", None)
        mock_llm.return_value = json.dumps(
            {
                "verdict": "no",
                "evaluation": "hallucination",
                "score": 0.6,  # Above 0.5 but below 0.7
                "classification": "factual_inaccuracy",
                "explanation": "Test",
            }
        )

        # With threshold 0.7, score of 0.6 should be "no"
        detector = Hallucination(provider="openai", threshold_score=0.7)
        result = detector.measure(prompt="Test", contexts=["Context"], text="Text")

        assert result.verdict == "no"
