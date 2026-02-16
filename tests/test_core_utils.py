"""
Tests for the tmam.core modules (datetime_utils, pydantic_utilities).
"""

import pytest
from datetime import datetime, date, timezone
from unittest.mock import MagicMock, patch

import sys

sys.path.insert(0, "/home/claude/sdk/sdk/src")


class TestDatetimeUtils:
    """Tests for datetime utility functions."""

    def test_import_datetime_utils(self):
        """Test that datetime_utils module can be imported."""
        from tmam.core import datetime_utils

        assert datetime_utils is not None


class TestPydanticUtilities:
    """Tests for Pydantic utility functions."""

    def test_import_pydantic_utilities(self):
        """Test that pydantic_utilities module can be imported."""
        from tmam.core import pydantic_utilities

        assert pydantic_utilities is not None


class TestUtilsModule:
    """Tests for the utils module."""

    def test_import_utils(self):
        """Test that utils module can be imported."""
        from tmam.utils import utils

        assert utils is not None

    def test_json_output_class(self):
        """Test JsonOutput class from utils."""
        from tmam.utils.utils import JsonOutput

        # JsonOutput should be importable
        assert JsonOutput is not None


class TestExperimentUtils:
    """Tests for experiment utility functions."""

    def test_import_experiment(self):
        """Test that experiment module can be imported."""
        from tmam.utils import experiment

        assert experiment is not None

    def test_run_async_safely_exists(self):
        """Test that run_async_safely function exists."""
        from tmam.utils.experiment import run_async_safely

        assert callable(run_async_safely)

    def test_run_evaluator_def_exists(self):
        """Test that run_evaluator_def function exists."""
        from tmam.utils.experiment import run_evaluator_def

        assert callable(run_evaluator_def)

    def test_run_async_safely_with_coroutine(self):
        """Test run_async_safely with a coroutine."""
        from tmam.utils.experiment import run_async_safely

        async def async_func():
            return 42

        result = run_async_safely(async_func())
        assert result == 42

    def test_format_value_short_string(self):
        """Test format_value with a short string."""
        from tmam.utils.experiment import format_value

        result = format_value("hello")
        assert result == "hello"

    def test_format_value_long_string(self):
        """Test format_value with a long string."""
        from tmam.utils.experiment import format_value

        long_string = "a" * 100
        result = format_value(long_string)
        assert len(result) == 53  # 50 chars + "..."
        assert result.endswith("...")

    def test_format_value_non_string(self):
        """Test format_value with non-string value."""
        from tmam.utils.experiment import format_value

        result = format_value(123)
        assert result == "123"
