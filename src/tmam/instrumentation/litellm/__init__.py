# pylint: disable=useless-return, bad-staticmethod-argument, disable=duplicate-code
"""Initializer of Auto Instrumentation of LiteLLM Functions"""

from typing import Collection
import importlib.metadata
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from wrapt import wrap_function_wrapper

from tmam.instrumentation.litellm.litellm import (
    completion, embedding
)
from tmam.instrumentation.litellm.async_litellm import (
   acompletion, aembedding
)

_instruments = ("litellm >= 1.52.6",)

class LiteLLMInstrumentor(BaseInstrumentor):
    """
    An instrumentor for LiteLLM's client library.
    """

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        application_name = kwargs.get("application_name", "default_application")
        environment = kwargs.get("environment", "default_environment")
        tracer = kwargs.get("tracer")
        metrics = kwargs.get("metrics_dict")
        pricing_info = kwargs.get("pricing_info", {})
        capture_message_content = kwargs.get("capture_message_content", False)
        disable_metrics = kwargs.get("disable_metrics")
        version = importlib.metadata.version("litellm")

        # completion
        wrap_function_wrapper(
            "litellm",
            "completion",
            completion(version, environment, application_name,
                  tracer, pricing_info, capture_message_content, metrics, disable_metrics),
        )

        wrap_function_wrapper(
            "litellm",
            "acompletion",
            acompletion(version, environment, application_name,
                  tracer, pricing_info, capture_message_content, metrics, disable_metrics),
        )

        wrap_function_wrapper(
            "litellm",
            "embedding",
            embedding(version, environment, application_name,
                  tracer, pricing_info, capture_message_content, metrics, disable_metrics),
        )

        wrap_function_wrapper(
            "litellm",
            "aembedding",
            aembedding(version, environment, application_name,
                  tracer, pricing_info, capture_message_content, metrics, disable_metrics),
        )

    def _uninstrument(self, **kwargs):
        # Proper uninstrumentation logic to revert patched methods
        pass
