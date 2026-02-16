"""
Tests for the tmam.model.dataset module.
"""

import pytest
from unittest.mock import MagicMock, patch
from typing import Dict, Any, List

import sys

sys.path.insert(0, "/home/claude/sdk/sdk/src")

from tmam.model.dataset import (
    ExperimentItem,
    ExperimentData,
    Evaluation,
    ExperimentItemResult,
    TaskFunction,
)


class TestDatasetModels:
    """Tests for dataset-related models."""

    def test_experiment_item(self):
        """Test ExperimentItem model."""
        item = ExperimentItem(
            input="What is 2+2?",
            expected_output="4",
            metadata={"difficulty": "easy"},
        )

        assert item.input == "What is 2+2?"
        assert item.expected_output == "4"
        assert item.metadata == {"difficulty": "easy"}

    def test_experiment_item_without_metadata(self):
        """Test ExperimentItem without metadata."""
        item = ExperimentItem(
            input="Test input",
            expected_output="Test output",
        )

        assert item.input == "Test input"
        assert item.expected_output == "Test output"
        assert item.metadata is None

    def test_experiment_data(self):
        """Test ExperimentData model."""
        items = [
            ExperimentItem(input="Q1", expected_output="A1"),
            ExperimentItem(input="Q2", expected_output="A2"),
        ]

        data = ExperimentData(items=items)

        assert len(data.items) == 2
        assert data.items[0].input == "Q1"
        assert data.items[1].expected_output == "A2"

    def test_evaluation_class(self):
        """Test Evaluation class."""
        eval_result = Evaluation(
            name="accuracy",
            value=0.95,
            comment="High accuracy achieved",
            metadata={"model": "gpt-4o"},
            config_id="config-123",
            data_type="numeric",
        )

        assert eval_result.name == "accuracy"
        assert eval_result.value == 0.95
        assert eval_result.comment == "High accuracy achieved"
        assert eval_result.metadata == {"model": "gpt-4o"}
        assert eval_result.config_id == "config-123"
        assert eval_result.data_type == "numeric"

    def test_evaluation_minimal(self):
        """Test Evaluation with minimal arguments."""
        eval_result = Evaluation(
            name="score",
            value=0.8,
        )

        assert eval_result.name == "score"
        assert eval_result.value == 0.8
        assert eval_result.comment is None
        assert eval_result.metadata is None

    def test_evaluation_with_string_value(self):
        """Test Evaluation with string value."""
        eval_result = Evaluation(
            name="category",
            value="positive",
            data_type="categorical",
        )

        assert eval_result.value == "positive"
        assert eval_result.data_type == "categorical"

    def test_evaluation_with_boolean_value(self):
        """Test Evaluation with boolean value."""
        eval_result = Evaluation(
            name="is_safe",
            value=True,
            data_type="boolean",
        )

        assert eval_result.value is True

    def test_evaluation_with_none_value(self):
        """Test Evaluation with None value (for failed evaluations)."""
        eval_result = Evaluation(
            name="failed_eval",
            value=None,
            comment="Could not compute score",
        )

        assert eval_result.value is None
        assert eval_result.comment == "Could not compute score"

    def test_experiment_item_result(self):
        """Test ExperimentItemResult class."""
        item = ExperimentItem(
            input="Test query",
            expected_output="Expected result",
        )

        evaluations = [Evaluation(name="accuracy", value=0.9, comment="Good")]

        result = ExperimentItemResult(
            item=item,
            output={"result": "Actual result"},
            evaluations=evaluations,
            trace_id="trace-123",
            dataset_run_id="run-456",
        )

        assert result.item is item
        assert result.output == {"result": "Actual result"}
        assert len(result.evaluations) == 1
        assert result.trace_id == "trace-123"
        assert result.dataset_run_id == "run-456"

    def test_experiment_item_result_with_none_ids(self):
        """Test ExperimentItemResult with None trace/run IDs."""
        item = ExperimentItem(
            input="Test",
            expected_output="Output",
        )

        result = ExperimentItemResult(
            item=item,
            output="Generated output",
            evaluations=[],
            trace_id=None,
            dataset_run_id=None,
        )

        assert result.trace_id is None
        assert result.dataset_run_id is None


class TestFunctionTypes:
    """Tests for function type definitions."""

    def test_task_function_protocol(self):
        """Test that functions can match the TaskFunction protocol."""

        def my_task(*, item: ExperimentItem, **kwargs) -> Dict[str, Any]:
            return {"result": f"Processed {item.input}"}

        # Verify the function works correctly
        item = ExperimentItem(
            input="Test query",
            expected_output="Expected",
        )

        result = my_task(item=item)
        assert "result" in result
        assert "Test query" in result["result"]

    def test_evaluator_function_returns_evaluation(self):
        """Test that evaluator functions can return Evaluation objects."""

        def my_evaluator(
            *, input: Any, output: Any, expected_output: Any = None, **kwargs
        ) -> Evaluation:
            score = 1.0 if output == expected_output else 0.0
            return Evaluation(
                name="exact_match",
                value=score,
                comment="Exact match evaluation",
            )

        result = my_evaluator(
            input={"query": "Test"},
            output="answer",
            expected_output="answer",
        )

        assert isinstance(result, Evaluation)
        assert result.value == 1.0

    def test_evaluator_returning_list(self):
        """Test evaluator that returns multiple Evaluations."""

        def multi_evaluator(*, input: Any, output: Any, **kwargs) -> List[Evaluation]:
            return [
                Evaluation(name="length", value=len(str(output))),
                Evaluation(name="has_content", value=bool(output)),
            ]

        results = multi_evaluator(
            input="Test",
            output="Some output",
        )

        assert len(results) == 2
        assert results[0].name == "length"
        assert results[1].name == "has_content"


class TestExperimentDataScenarios:
    """Tests for various experiment data scenarios."""

    def test_empty_experiment_data(self):
        """Test ExperimentData with no items."""
        data = ExperimentData(items=[])
        assert len(data.items) == 0

    def test_experiment_data_with_many_items(self):
        """Test ExperimentData with many items."""
        items = [
            ExperimentItem(input=f"Q{i}", expected_output=f"A{i}") for i in range(100)
        ]

        data = ExperimentData(items=items)

        assert len(data.items) == 100
        assert data.items[0].input == "Q0"
        assert data.items[99].expected_output == "A99"

    def test_experiment_item_with_special_characters(self):
        """Test ExperimentItem with special characters in content."""
        item = ExperimentItem(
            input="What is 2+2? ñ é ü 中文",
            expected_output="4 with special: ñ é ü 中文",
            metadata={"language": "mixed"},
        )

        assert "ñ" in item.input
        assert "中文" in item.expected_output

    def test_experiment_item_with_long_content(self):
        """Test ExperimentItem with long content."""
        long_text = "A" * 10000

        item = ExperimentItem(
            input=long_text,
            expected_output=long_text,
        )

        assert len(item.input) == 10000
        assert len(item.expected_output) == 10000


class TestEvaluationEdgeCases:
    """Tests for Evaluation edge cases."""

    def test_evaluation_with_zero_value(self):
        """Test Evaluation with zero as value."""
        eval_result = Evaluation(name="score", value=0)
        assert eval_result.value == 0

    def test_evaluation_with_negative_value(self):
        """Test Evaluation with negative value."""
        eval_result = Evaluation(name="delta", value=-0.5)
        assert eval_result.value == -0.5

    def test_evaluation_with_empty_string_comment(self):
        """Test Evaluation with empty string comment."""
        eval_result = Evaluation(name="score", value=0.5, comment="")
        assert eval_result.comment == ""

    def test_evaluation_with_complex_metadata(self):
        """Test Evaluation with complex nested metadata."""
        metadata = {
            "model": "gpt-4",
            "details": {
                "temperature": 0.7,
                "tokens": {"input": 100, "output": 50},
            },
            "tags": ["test", "evaluation"],
        }

        eval_result = Evaluation(
            name="score",
            value=0.85,
            metadata=metadata,
        )

        assert eval_result.metadata["details"]["tokens"]["input"] == 100


class TestExperimentItemResultWithMultipleEvaluations:
    """Tests for ExperimentItemResult with multiple evaluations."""

    def test_multiple_evaluations(self):
        """Test result with multiple evaluations."""
        item = ExperimentItem(input="Q", expected_output="A")

        evaluations = [
            Evaluation(name="accuracy", value=0.9),
            Evaluation(name="relevance", value=0.85),
            Evaluation(name="fluency", value=0.95),
        ]

        result = ExperimentItemResult(
            item=item,
            output="Generated A",
            evaluations=evaluations,
            trace_id="trace-1",
            dataset_run_id=None,
        )

        assert len(result.evaluations) == 3
        assert result.evaluations[0].name == "accuracy"
        assert result.evaluations[1].name == "relevance"
        assert result.evaluations[2].name == "fluency"
