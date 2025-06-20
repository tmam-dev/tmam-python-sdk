# pylint: disable=useless-return, bad-staticmethod-argument, disable=duplicate-code
"""Initializer of Auto Instrumentation of Mistral Functions"""
from typing import Collection
import importlib.metadata
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from wrapt import wrap_function_wrapper

from tmam.instrumentation.mistral.mistral import chat, chat_stream, embeddings
from tmam.instrumentation.mistral.async_mistral import async_chat, async_chat_stream
from tmam.instrumentation.mistral.async_mistral import async_embeddings

_instruments = ("mistralai >= 1.0.0",)

class MistralInstrumentor(BaseInstrumentor):
    """An instrumentor for Mistral's client library."""

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        application_name = kwargs.get("application_name")
        environment = kwargs.get("environment")
        tracer = kwargs.get("tracer")
        metrics = kwargs.get("metrics_dict")
        pricing_info = kwargs.get("pricing_info")
        capture_message_content = kwargs.get("capture_message_content")
        disable_metrics = kwargs.get("disable_metrics")
        version = importlib.metadata.version("mistralai")

        # sync
        wrap_function_wrapper(
            "mistralai.chat",  
            "Chat.complete",  
            chat(version, environment, application_name,
                 tracer, pricing_info, capture_message_content, metrics, disable_metrics),
        )

        # sync
        wrap_function_wrapper(
            "mistralai.chat",  
            "Chat.stream",  
            chat_stream(version, environment, application_name,
                        tracer, pricing_info, capture_message_content, metrics, disable_metrics),
        )

        # sync
        wrap_function_wrapper(
            "mistralai.embeddings",  
            "Embeddings.create",  
            embeddings(version, environment, application_name,
                       tracer, pricing_info, capture_message_content, metrics, disable_metrics),
        )

        # Async
        wrap_function_wrapper(
            "mistralai.chat",  
            "Chat.complete_async",  
            async_chat(version, environment, application_name,
                       tracer, pricing_info, capture_message_content, metrics, disable_metrics),
        )

        # Async
        wrap_function_wrapper(
            "mistralai.chat",  
            "Chat.stream_async",  
            async_chat_stream(version, environment, application_name,
                              tracer, pricing_info, capture_message_content, metrics, disable_metrics),
        )

        #sync
        wrap_function_wrapper(
            "mistralai.embeddings",  
            "Embeddings.create_async",  
            async_embeddings(version, environment, application_name,
                             tracer, pricing_info, capture_message_content, metrics, disable_metrics),
        )

    @staticmethod
    def _uninstrument(self, **kwargs):
        pass
