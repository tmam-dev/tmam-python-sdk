# pylint: disable=broad-exception-caught
"""
The __init__.py module for the Tmam package.
This module sets up the Tmam configuration and instrumentation for various
large language models (LLMs).
"""

import asyncio
from typing import Any, Callable, Dict, List, Optional, TypedDict, Union, cast
import logging
import os
from importlib.util import find_spec
from functools import wraps
from contextlib import contextmanager
import requests
import datetime as dt


# Import internal modules for setting up tracing and fetching pricing info.
from opentelemetry import trace as t
from opentelemetry.trace import SpanKind, Status, StatusCode, Span
from opentelemetry.sdk.resources import SERVICE_NAME, DEPLOYMENT_ENVIRONMENT
from tmam.model.dataset import (
    CreateDatasetRunItemRequest,
    DatasetModel,
    Evaluation,
    EvaluatorFunction,
    ExperimentData,
    ExperimentItem,
    ExperimentItemResult,
    ExperimentResult,
    RunEvaluatorFunction,
    TaskFunction,
)
from tmam.utils.experiment import run_async_safely, run_evaluator_def
from tmam.utils.utils import JsonOutput
from tmam.semcov import SemanticConvetion
from tmam.otel.tracing import setup_tracing
from tmam.otel.metrics import setup_meter
from tmam.otel.events import setup_events
from tmam.__helpers import fetch_pricing_info, get_env_variable

# Instrumentors for various large language models.
from tmam.instrumentation.openai import OpenAIInstrumentor
from tmam.instrumentation.anthropic import AnthropicInstrumentor
from tmam.instrumentation.cohere import CohereInstrumentor
from tmam.instrumentation.mistral import MistralInstrumentor
from tmam.instrumentation.bedrock import BedrockInstrumentor
from tmam.instrumentation.vertexai import VertexAIInstrumentor
from tmam.instrumentation.groq import GroqInstrumentor
from tmam.instrumentation.ollama import OllamaInstrumentor
from tmam.instrumentation.gpt4all import GPT4AllInstrumentor
from tmam.instrumentation.elevenlabs import ElevenLabsInstrumentor
from tmam.instrumentation.vllm import VLLMInstrumentor
from tmam.instrumentation.google_ai_studio import GoogleAIStudioInstrumentor
from tmam.instrumentation.reka import RekaInstrumentor
from tmam.instrumentation.premai import PremAIInstrumentor
from tmam.instrumentation.assemblyai import AssemblyAIInstrumentor
from tmam.instrumentation.azure_ai_inference import AzureAIInferenceInstrumentor
from tmam.instrumentation.langchain import LangChainInstrumentor
from tmam.instrumentation.llamaindex import LlamaIndexInstrumentor
from tmam.instrumentation.haystack import HaystackInstrumentor
from tmam.instrumentation.embedchain import EmbedChainInstrumentor
from tmam.instrumentation.mem0 import Mem0Instrumentor
from tmam.instrumentation.chroma import ChromaInstrumentor
from tmam.instrumentation.pinecone import PineconeInstrumentor
from tmam.instrumentation.qdrant import QdrantInstrumentor
from tmam.instrumentation.milvus import MilvusInstrumentor
from tmam.instrumentation.astra import AstraInstrumentor
from tmam.instrumentation.transformers import TransformersInstrumentor
from tmam.instrumentation.litellm import LiteLLMInstrumentor
from tmam.instrumentation.together import TogetherInstrumentor
from tmam.instrumentation.crewai import CrewAIInstrumentor
from tmam.instrumentation.ag2 import AG2Instrumentor
from tmam.instrumentation.multion import MultiOnInstrumentor
from tmam.instrumentation.dynamiq import DynamiqInstrumentor
from tmam.instrumentation.phidata import PhidataInstrumentor
from tmam.instrumentation.julep import JulepInstrumentor
from tmam.instrumentation.ai21 import AI21Instrumentor
from tmam.instrumentation.controlflow import ControlFlowInstrumentor
from tmam.instrumentation.crawl4ai import Crawl4AIInstrumentor
from tmam.instrumentation.firecrawl import FireCrawlInstrumentor
from tmam.instrumentation.letta import LettaInstrumentor
from tmam.instrumentation.openai_agents import OpenAIAgentsInstrumentor
from tmam.instrumentation.gpu import GPUInstrumentor
import tmam.evals

# Set up logging for error and information messages.
logger = logging.getLogger(__name__)


class TmamConfig:
    """
    A Singleton Configuration class for Tmam.

    This class maintains a single instance of configuration settings including
    environment details, application name, and tracing information throughout the Tmam package.

    Attributes:
        environment (str): Deployment environment of the application.
        application_name (str): Name of the application using Tmam.
        pricing_info (Dict[str, Any]): Pricing information.
        tracer (Optional[Any]): Tracer instance for OpenTelemetry.
        event_provider (Optional[Any]): Event logger provider for OpenTelemetry.
        disable_batch (bool): Flag to disable batch span processing in tracing.
        capture_message_content (bool): Flag to enable or disable tracing of content.
    """

    _instance = None

    def __new__(cls):
        """Ensures that only one instance of the configuration exists."""
        if cls._instance is None:
            cls._instance = super(TmamConfig, cls).__new__(cls)
            cls.reset_to_defaults()
        return cls._instance

    @classmethod
    def reset_to_defaults(cls):
        """Resets configuration to default values."""
        cls.environment = "default"
        cls.application_name = "default"
        cls.pricing_info = {}
        cls.tracer = None
        cls.event_provider = None
        cls.metrics_dict = {}
        cls.disable_batch = False
        cls.capture_message_content = True
        cls.disable_metrics = False
        cls.url = None
        cls.public_key = None
        cls.secrect_key = None
        cls.last_guard_prompt_id = None
        cls.guardrail_id = None
        cls.name = None
        cls.user_id = None

    @classmethod
    def update_config(
        cls,
        environment,
        application_name,
        tracer,
        event_provider,
        disable_batch,
        capture_message_content,
        metrics_dict,
        disable_metrics,
        pricing_json,
        url,
        public_key,
        secrect_key,
        last_guard_prompt_id,
        guardrail_id,
        name,
        user_id,
    ):
        """
        Updates the configuration based on provided parameters.

        Args:
            environment (str): Deployment environment.
            application_name (str): Application name.
            tracer: Tracer instance.
            event_provider: Event logger provider instance.
            meter: Metric Instance
            disable_batch (bool): Disable batch span processing flag.
            capture_message_content (bool): Enable or disable content tracing.
            metrics_dict: Dictionary of metrics.
            disable_metrics (bool): Flag to disable metrics.
            pricing_json(str): path or url to the pricing json file
        """
        cls.environment = environment
        cls.application_name = application_name
        cls.pricing_info = fetch_pricing_info(pricing_json)
        cls.tracer = tracer
        cls.event_provider = event_provider
        cls.metrics_dict = metrics_dict
        cls.disable_batch = disable_batch
        cls.capture_message_content = capture_message_content
        cls.disable_metrics = disable_metrics
        cls.url = url
        cls.public_key = public_key
        cls.secrect_key = secrect_key
        cls.last_guard_prompt_id = last_guard_prompt_id
        cls.guardrail_id = guardrail_id
        cls.name = name
        cls.user_id = user_id

    @classmethod
    def update_guard_config(
        cls, last_guard_prompt_id, name, user_id, guardrail_id: Optional[str]
    ):
        """
        Updates the configuration based on provided parameters.

        Args:
        """
        cls.last_guard_prompt_id = last_guard_prompt_id
        cls.name = name
        cls.user_id = user_id
        if guardrail_id is not None:
            cls.guardrail_id = guardrail_id


def module_exists(module_name):
    """Check if nested modules exist, addressing the dot notation issue."""
    parts = module_name.split(".")
    for i in range(1, len(parts) + 1):
        if find_spec(".".join(parts[:i])) is None:
            return False
    return True


def instrument_if_available(
    instrumentor_name,
    instrumentor_instance,
    config,
    disabled_instrumentors,
    module_name_map,
):
    """Instruments the specified instrumentor if its library is available."""
    if instrumentor_name in disabled_instrumentors:
        logger.info("Instrumentor %s is disabled", instrumentor_name)
        return

    module_name = module_name_map.get(instrumentor_name)

    if not module_name:
        logger.error("No module mapping for %s", instrumentor_name)
        return

    try:
        if module_exists(module_name):
            instrumentor_instance.instrument(
                environment=config.environment,
                application_name=config.application_name,
                tracer=config.tracer,
                event_provider=config.event_provider,
                pricing_info=config.pricing_info,
                capture_message_content=config.capture_message_content,
                metrics_dict=config.metrics_dict,
                disable_metrics=config.disable_metrics,
            )
        else:
            # pylint: disable=line-too-long
            logger.info(
                "Library for %s (%s) not found. Skipping instrumentation",
                instrumentor_name,
                module_name,
            )
    except Exception as e:
        logger.error("Failed to instrument %s: %s", instrumentor_name, e)


def init(
    environment="default",
    application_name="default",
    tracer=None,
    event_logger=None,
    url=None,
    public_key=None,
    secrect_key=None,
    disable_batch=False,
    capture_message_content=True,
    disabled_instrumentors=None,
    meter=None,
    disable_metrics=False,
    pricing_json=None,
    collect_gpu_stats=False,
    guardrail_id=None,
):
    """
    Initializes the Tmam configuration and setups tracing.

    This function sets up the Tmam environment with provided configurations
    and initializes instrumentors for tracing.

    Args:
        environment (str): Deployment environment.
        application_name (str): Application name.
        tracer: Tracer instance (Optional).
        event_logger: EventLoggerProvider instance (Optional).
        meter: OpenTelemetry Metrics Instance (Optional).
        url (str): OTLP url for exporter.
        public_key (str): OTLP public key.
        secrect_key (str): OTLP secrect key.
        disable_batch (bool): Flag to disable batch span processing (Optional).
        capture_message_content (bool): Flag to trace content (Optional).
        disabled_instrumentors (List[str]): Optional. List of instrumentor names to disable.
        disable_metrics (bool): Flag to disable metrics (Optional).
        pricing_json(str): File path or url to the pricing json (Optional).
        collect_gpu_stats (bool): Flag to enable or disable GPU metrics collection.
    """
    disabled_instrumentors = disabled_instrumentors if disabled_instrumentors else []
    logger.info("Starting Tmam initialization...")

    module_name_map = {
        "openai": "openai",
        "anthropic": "anthropic",
        "cohere": "cohere",
        "mistral": "mistralai",
        "bedrock": "boto3",
        "vertexai": "vertexai",
        "groq": "groq",
        "ollama": "ollama",
        "gpt4all": "gpt4all",
        "elevenlabs": "elevenlabs",
        "vllm": "vllm",
        "google-ai-studio": "google.genai",
        "azure-ai-inference": "azure.ai.inference",
        "langchain": "langchain",
        "llama_index": "llama_index",
        "haystack": "haystack",
        "embedchain": "embedchain",
        "mem0": "mem0",
        "chroma": "chromadb",
        "pinecone": "pinecone",
        "qdrant": "qdrant_client",
        "milvus": "pymilvus",
        "transformers": "transformers",
        "litellm": "litellm",
        "crewai": "crewai",
        "ag2": "ag2",
        "autogen": "autogen",
        "pyautogen": "pyautogen",
        "multion": "multion",
        "dynamiq": "dynamiq",
        "phidata": "phi",
        "reka-api": "reka",
        "premai": "premai",
        "julep": "julep",
        "astra": "astrapy",
        "ai21": "ai21",
        "controlflow": "controlflow",
        "assemblyai": "assemblyai",
        "crawl4ai": "crawl4ai",
        "firecrawl": "firecrawl",
        "letta": "letta",
        "together": "together",
        "openai-agents": "agents",
    }

    invalid_instrumentors = [
        name for name in disabled_instrumentors if name not in module_name_map
    ]
    for invalid_name in invalid_instrumentors:
        logger.warning(
            "Invalid instrumentor name detected and ignored: '%s'", invalid_name
        )

    # Validate and set the base URL
    env_url = get_env_variable(
        "TMAM_URL",
        url,
        "Missing Tmam URL: Provide as arg or set TMAM_URL env var.",
    )

    # Validate and set the API key
    env_pk_key = get_env_variable(
        "TMAM_PUBLIC_KEY",
        public_key,
        "Missing Public key: Provide as arg or set TMAM_PUBLIC_KEY env var.",
    )
    env_sk_key = get_env_variable(
        "TMAM_SECRET_KEY",
        secrect_key,
        "Missing Secret key: Provide as arg or set TMAM_SECRET_KEY env var.",
    )

    try:
        # Retrieve or create the single configuration instance.
        config = TmamConfig()

        env_url_p = env_url.replace("v1", "")

        # Setup tracing based on the provided or default configuration.
        tracer = setup_tracing(
            application_name=application_name,
            environment=environment,
            tracer=tracer,
            url=env_url_p,
            public_key=env_pk_key,
            secrect_key=env_sk_key,
            disable_batch=disable_batch,
        )

        if not tracer:
            logger.error("Tmam tracing setup failed. Tracing will not be available.")
            return

        # Setup events based on the provided or default configuration.
        event_provider = setup_events(
            application_name=application_name,
            environment=environment,
            event_logger=event_logger,
            url=env_url_p,
            public_key=env_pk_key,
            secrect_key=env_sk_key,
            disable_batch=disable_batch,
        )

        if not event_provider:
            logger.error("Tmam events setup failed. Events will not be available")

        # Setup meter and receive metrics_dict instead of meter.
        metrics_dict, err = setup_meter(
            application_name=application_name,
            environment=environment,
            meter=meter,
            url=env_url_p,
            public_key=env_pk_key,
            secrect_key=env_sk_key,
        )

        if err:
            logger.error(
                "Tmam metrics setup failed. Metrics will not be available: %s", err
            )
            return

        if (
            os.getenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "").lower
            == "false"
        ):
            capture_message_content = False

        # Update global configuration with the provided settings.
        config.update_config(
            environment=environment,
            application_name=application_name,
            tracer=tracer,
            event_provider=event_provider,
            disable_batch=disable_batch,
            capture_message_content=capture_message_content,
            metrics_dict=metrics_dict,
            disable_metrics=disable_metrics,
            pricing_json=pricing_json,
            url=url,
            public_key=public_key,
            secrect_key=secrect_key,
            last_guard_prompt_id=None,
            guardrail_id=guardrail_id,
            name=None,
            user_id=None,
        )
        # Map instrumentor names to their instances
        instrumentor_instances = {
            "openai": OpenAIInstrumentor(),
            "anthropic": AnthropicInstrumentor(),
            "cohere": CohereInstrumentor(),
            "mistral": MistralInstrumentor(),
            "bedrock": BedrockInstrumentor(),
            "vertexai": VertexAIInstrumentor(),
            "groq": GroqInstrumentor(),
            "ollama": OllamaInstrumentor(),
            "gpt4all": GPT4AllInstrumentor(),
            "elevenlabs": ElevenLabsInstrumentor(),
            "vllm": VLLMInstrumentor(),
            "google-ai-studio": GoogleAIStudioInstrumentor(),
            "azure-ai-inference": AzureAIInferenceInstrumentor(),
            "langchain": LangChainInstrumentor(),
            "llama_index": LlamaIndexInstrumentor(),
            "haystack": HaystackInstrumentor(),
            "embedchain": EmbedChainInstrumentor(),
            "mem0": Mem0Instrumentor(),
            "chroma": ChromaInstrumentor(),
            "pinecone": PineconeInstrumentor(),
            "qdrant": QdrantInstrumentor(),
            "milvus": MilvusInstrumentor(),
            "transformers": TransformersInstrumentor(),
            "litellm": LiteLLMInstrumentor(),
            "crewai": CrewAIInstrumentor(),
            "ag2": AG2Instrumentor(),
            "multion": MultiOnInstrumentor(),
            "autogen": AG2Instrumentor(),
            "pyautogen": AG2Instrumentor(),
            "dynamiq": DynamiqInstrumentor(),
            "phidata": PhidataInstrumentor(),
            "reka-api": RekaInstrumentor(),
            "premai": PremAIInstrumentor(),
            "julep": JulepInstrumentor(),
            "astra": AstraInstrumentor(),
            "ai21": AI21Instrumentor(),
            "controlflow": ControlFlowInstrumentor(),
            "assemblyai": AssemblyAIInstrumentor(),
            "crawl4ai": Crawl4AIInstrumentor(),
            "firecrawl": FireCrawlInstrumentor(),
            "letta": LettaInstrumentor(),
            "together": TogetherInstrumentor(),
            "openai-agents": OpenAIAgentsInstrumentor(),
        }

        # Initialize and instrument only the enabled instrumentors
        for name, instrumentor in instrumentor_instances.items():
            instrument_if_available(
                name, instrumentor, config, disabled_instrumentors, module_name_map
            )

        if not disable_metrics and collect_gpu_stats:
            GPUInstrumentor().instrument(
                environment=config.environment,
                application_name=config.application_name,
            )
    except Exception as e:
        logger.error("Error during Tmam initialization: %s", e)


def get_prompt(
    name=None,
    prompt_id=None,
    label=None,
    version=None,
):
    """
    Retrieve and returns the prompt from Tmam Prompt Hub
    """

    config = TmamConfig()

    if config.url is None or config.public_key is None or config.secrect_key is None:
        raise ValueError("make sure tmam.init is defined")

    # Construct the API endpoint
    endpoint = config.url + "/prompt/compiled"

    # Prepare the payload
    payload = {
        "name": name,
        "promptId": prompt_id,
        "version": version,
        "label": label,
        "source": "Python",
    }

    # Remove None values from payload
    payload = {k: v for k, v in payload.items() if v is not None}

    # Prepare headers
    headers = {
        "X-Public-Key": config.public_key,
        "X-Secret-Key": config.secrect_key,
        "Content-Type": "application/json",
    }

    try:
        # Make the POST request to the API with headers
        response = requests.post(endpoint, json=payload, headers=headers, timeout=120)

        # Check if the response is successful
        response.raise_for_status()

        # Return the JSON response
        return response.json()["data"]
    except requests.RequestException as error:
        logger.error("Error fetching prompt: '%s'", error)
        return None


def get_secrets(
    key=None,
    tags=None,
    should_set_env=None,
):
    """
    Retrieve & returns the secrets from Tmam Vault & sets all to env is should_set_env is True
    """

    config = TmamConfig()

    if config.url is None or config.public_key is None or config.secrect_key is None:
        raise ValueError("make sure tmam.init is defined")

    # Construct the API endpoint
    endpoint = config.url + "/vault/secrets"

    # Prepare the payload
    payload = {"key": key, "tags": tags, "source": "Python"}

    # Remove None values from payload
    payload = {k: v for k, v in payload.items() if v is not None}

    # Prepare headers
    headers = {
        "X-Public-Key": config.public_key,
        "X-Secret-Key": config.secrect_key,
        "Content-Type": "application/json",
    }

    try:
        # Make the POST request to the API with headers
        response = requests.post(endpoint, json=payload, headers=headers, timeout=120)

        # Check if the response is successful
        response.raise_for_status()

        # Return the JSON response
        vault_response = response.json()["data"]

        res = vault_response.get("res", [])

        if should_set_env is True:
            for token, value in res.items():
                os.environ[token] = str(value)
        return vault_response
    except requests.RequestException as error:
        logger.error("Error fetching secrets: '%s'", error)
        return None


class AgentSpanRole:
    AGENT_RUN = "agent.run"
    PROMPT_BUILD = "agent.prompt.build"
    REASONING_STEP = "agent.reasoning.step"
    LLM_CALL = "llm.call"
    TOOL_CALL = "tool.call"
    MEMORY_RETRIEVE = "memory.retrieve"
    EXTERNAL_API = "external.api.call"


class AISpanAttributes:
    # Identity
    SPAN_ROLE = "ai.span.role"
    COMPONENT = "ai.component"  # llm | agent | tool | memory | api

    # LLM-specific
    LLM_COMPLETION = SemanticConvetion.GEN_AI_CONTENT_COMPLETION
    LLM_MODEL = "llm.model"
    LLM_PROMPT_TOKENS = "llm.prompt.tokens"
    LLM_COMPLETION_TOKENS = "llm.completion.tokens"
    LLM_TOTAL_TOKENS = "llm.total.tokens"

    # Agent-specific
    AGENT_STEP_INDEX = "agent.step.index"
    AGENT_ITERATION = "agent.iteration"
    AGENT_DECISION = "agent.decision"

    # Tool / memory / api
    TOOL_NAME = "tool.name"
    MEMORY_BACKEND = "memory.backend"
    API_ENDPOINT = "api.endpoint"


# def trace(wrapped):
#     """
#     Generates a telemetry wrapper for messages to collect metrics.
#     """
#     if not callable(wrapped):
#         raise TypeError(
#             f"@trace can only be applied to callable objects, got {type(wrapped).__name__}"
#         )

#     try:
#         __trace = t.get_tracer_provider()
#         tracer = __trace.get_tracer(__name__)
#     except Exception as tracer_exception:
#         logging.error(
#             "Failed to initialize tracer: %s", tracer_exception, exc_info=True
#         )
#         raise

#     @wraps(wrapped)
#     def wrapper(*args, **kwargs):
#         with tracer.start_as_current_span(
#             name=wrapped.__name__,
#             kind=SpanKind.CLIENT,
#         ) as span:
#             response = None
#             try:
#                 response = wrapped(*args, **kwargs)
#                 span.set_attribute(
#                     SemanticConvetion.GEN_AI_CONTENT_COMPLETION, response or ""
#                 )
#                 span.set_status(Status(StatusCode.OK))
#             except Exception as e:
#                 span.record_exception(e)
#                 span.set_status(status=Status(StatusCode.ERROR), description=str(e))
#                 logging.error("Error in %s: %s", wrapped.__name__, e, exc_info=True)
#                 raise

#             try:
#                 span.set_attribute("function.args", str(args))
#                 span.set_attribute("function.kwargs", str(kwargs))
#                 span.set_attribute(
#                     SERVICE_NAME,
#                     TmamConfig.application_name,
#                 )
#                 span.set_attribute(DEPLOYMENT_ENVIRONMENT, TmamConfig.environment)
#             except Exception as meta_exception:
#                 logging.error(
#                     "Failed to set metadata for %s: %s",
#                     wrapped.__name__,
#                     meta_exception,
#                     exc_info=True,
#                 )

#             return response

#     return wrapper


# class TracedSpan:
#     """
#     A wrapper class for an OpenTelemetry span that provides helper methods
#     for setting result and metadata attributes on the span.

#     Attributes:
#         _span (Span): The underlying OpenTelemetry span.
#     """

#     def __init__(self, span):
#         """
#         Initializes the TracedSpan with the given span.

#         Params:
#             span (Span): The OpenTelemetry span to be wrapped.
#         """

#         self._span: Span = span

#     def set_result(self, result):
#         """
#         Sets the result attribute on the underlying span.

#         Params:
#             result: The result to be set as an attribute on the span.
#         """

#         self._span.set_attribute(SemanticConvetion.GEN_AI_CONTENT_COMPLETION, result)

#     def set_metadata(self, metadata: Dict):
#         """
#         Sets multiple attributes on the underlying span.

#         Params:
#             metadata (Dict): A dictionary of attributes to be set on the span.
#         """

#         self._span.set_attributes(attributes=metadata)

#     def __enter__(self):
#         """
#         Enters the context of the TracedSpan, returning itself.

#         Returns:
#             TracedSpan: The instance of TracedSpan.
#         """

#         return self

#     def __exit__(self, _exc_type, _exc_val, _exc_tb):
#         """
#         Exits the context of the TracedSpan by ending the underlying span.
#         """

#         self._span.end()


# @contextmanager
# def start_trace(name: str):
#     """
#     A context manager that starts a new trace and provides a TracedSpan
#     for usage within the context.

#     Params:
#         name (str): The name of the span.

#     Yields:
#         TracedSpan: The wrapped span for trace operations.
#     """

#     __trace = t.get_tracer_provider()
#     with __trace.get_tracer(__name__).start_as_current_span(
#         name,
#         kind=SpanKind.CLIENT,
#     ) as span:
#         yield TracedSpan(span)

# -----------------------------
# AI span classification
# -----------------------------


class AISpanKind:
    LLM = "llm"
    AGENT = "agent"
    TOOL = "tool"
    MEMORY = "memory"
    EXTERNAL = "external"
    FUNCTION = "function"  # non-AI fallback


def classify_span(role: Optional[str]) -> str:
    if not role:
        return AISpanKind.FUNCTION

    r = role.lower()
    if "llm" in r:
        return AISpanKind.LLM
    if "agent" in r:
        return AISpanKind.AGENT
    if "tool" in r:
        return AISpanKind.TOOL
    if "memory" in r:
        return AISpanKind.MEMORY
    if "api" in r or "external" in r:
        return AISpanKind.EXTERNAL

    return AISpanKind.FUNCTION


def _safe(v: Any):
    if isinstance(v, (int, float, bool, str)):
        return v
    return str(v)


class TracedSpan:
    """
    Unified span wrapper for Agent + LLM tracing.
    """

    def __init__(self, span: Span, *, role: Optional[str], kind: str):
        self._span = span
        self._role = role
        self._kind = kind

        self._span.set_attribute("ai.span.kind", kind)
        self._span.set_attribute("ai.span.role", role or "function")

    # ---------- result handling ----------

    def set_result(self, result: Any):
        if result is None:
            return

        # Backward compatibility (old dashboards)
        self._span.set_attribute(
            "gen_ai.content.completion",
            str(result),
        )

        if self._kind == AISpanKind.LLM:
            self._span.set_attribute("ai.llm.completion", str(result))
        else:
            self._span.set_attribute("ai.result", str(result))

    # ---------- metadata ----------

    def set_metadata(self, metadata: Dict[str, Any]):
        if not metadata:
            return

        safe = {k: _safe(v) for k, v in metadata.items()}
        self._span.set_attributes(safe)

    # ---------- LLM helpers ----------

    def set_llm_usage(
        self,
        *,
        model: Optional[str] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
    ):
        if model:
            self._span.set_attribute("ai.llm.model", model)

        if prompt_tokens is not None:
            self._span.set_attribute("ai.llm.prompt_tokens", prompt_tokens)

        if completion_tokens is not None:
            self._span.set_attribute(
                "ai.llm.completion_tokens",
                completion_tokens,
            )

        if prompt_tokens is not None and completion_tokens is not None:
            self._span.set_attribute(
                "ai.llm.total_tokens",
                prompt_tokens + completion_tokens,
            )

    # ---------- Agent helpers ----------

    def set_agent_step(self, index: int):
        self._span.set_attribute("ai.agent.step_index", index)

    # ---------- context ----------

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val:
            self._span.record_exception(exc_val)
            self._span.set_status(Status(StatusCode.ERROR, str(exc_val)))
        # IMPORTANT: do NOT call span.end()


def trace(_func: Callable = None, *, role: Optional[str] = None):
    """
    Unified decorator for:
    - LLM calls
    - Agent runs / steps
    - Tool / memory / API calls
    - Normal functions
    """

    def decorator(wrapped: Callable):
        if not callable(wrapped):
            raise TypeError("@trace can only be applied to callables")

        tracer = t.get_tracer_provider().get_tracer(__name__)
        span_kind = classify_span(role)

        @wraps(wrapped)
        def wrapper(*args, **kwargs):
            span_name = role or wrapped.__name__

            with tracer.start_as_current_span(
                name=span_name,
                kind=SpanKind.CLIENT,
            ) as span:

                if not span.is_recording():
                    return wrapped(*args, **kwargs)

                # ---- canonical classification ----
                span.set_attribute("ai.span.kind", span_kind)
                span.set_attribute("ai.span.role", role or "function")

                # ---- code identity ----
                span.set_attribute("code.function", wrapped.__name__)

                # ---- env identity ----
                span.set_attribute("service.name", TmamConfig.application_name)
                span.set_attribute(
                    "deployment.environment",
                    TmamConfig.environment,
                )

                try:
                    result = wrapped(*args, **kwargs)

                    # legacy + new result attributes
                    span.set_attribute(
                        "gen_ai.content.completion",
                        _safe(result),
                    )

                    if span_kind == AISpanKind.LLM:
                        span.set_attribute("ai.llm.completion", _safe(result))
                    else:
                        span.set_attribute("ai.result", _safe(result))

                    span.set_status(Status(StatusCode.OK))
                    return result

                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

                finally:
                    # safe args capture (never fails exporter)
                    span.set_attribute("function.args", _safe(args))
                    span.set_attribute("function.kwargs", _safe(kwargs))

        return wrapper

    if _func is None:
        return decorator

    return decorator(_func)


@contextmanager
def start_trace(name: str, *, role: Optional[str] = None):
    """
    Manual span for:
    - agent reasoning loops
    - prompt building
    - tool / memory / API calls
    """

    tracer = t.get_tracer_provider().get_tracer(__name__)
    span_kind = classify_span(role)

    with tracer.start_as_current_span(
        name=name,
        kind=SpanKind.CLIENT,
    ) as span:

        if not span.is_recording():
            yield None
            return

        span.set_attribute("ai.span.kind", span_kind)
        span.set_attribute("ai.span.role", role or "function")
        span.set_attribute("code.span_name", name)

        span.set_attribute("service.name", TmamConfig.application_name)
        span.set_attribute(
            "deployment.environment",
            TmamConfig.environment,
        )

        yield TracedSpan(span, role=role, kind=span_kind)


class Detect:
    """
    A comprehensive class to detect prompt injections, valid/invalid topics, and sensitive topics using LLM or custom rules.

    Attributes:
        input (Optional[str]): The name of the LLM provider.
        output (Optional[str]): The API key for authenticating with the LLM.
    """

    def input(
        self,
        text: str,
        guardrail_id: str | None = None,
        name: str | None = None,
        user_id: str | None = None,
    ) -> JsonOutput:
        """
        Retrieve and returns the result from Tmam Guardrail.

        Args:
            text (str): text.
            guardrail_id Optional[str]: The guardrail ID for authenticating with the server, If not entered guardrail ID default guardrail will assigned.
            name (Optional[str]): The name of the guardrail for indentify purposes.
            user_id (Optional[str]): The user id of your prompt user.
        """

        config = TmamConfig()

        gid = guardrail_id if guardrail_id is not None else config.guardrail_id

        if (
            config.url is None
            or config.public_key is None
            or config.secrect_key is None
        ):
            raise ValueError("make sure tmam.init is defined")

        endpoint = config.url + "/guardrail/detect"

        payload = {
            "guardrailId": gid,
            "promptUserId": user_id,
            "prompt": text,
            "isInput": True,
            # "guardPromptId": Null
        }

        # Prepare headers
        headers = {
            "X-Public-Key": config.public_key,
            "X-Secret-Key": config.secrect_key,
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                endpoint, json=payload, headers=headers, timeout=120
            )
            response.raise_for_status()
            json = response.json()

            config.update_guard_config(
                last_guard_prompt_id=json["data"]["guardPromptId"],
                guardrail_id=gid,
                name=name,
                user_id=user_id,
            )

            return json["data"]["result"]
        except requests.RequestException as error:
            return error

    def output(self, text: str) -> JsonOutput:
        """
        Retrieve and returns the result from Tmam Guardrail.

        Args:
            text (str): text.
        """

        config = TmamConfig()

        if config.last_guard_prompt_id is None:
            raise ValueError("make sure input is defined")

        if (
            config.url is None
            or config.public_key is None
            or config.secrect_key is None
        ):
            raise ValueError("make sure tmam.init is defined")

        endpoint = config.url + "/guardrail/detect"

        usrid = config.user_id

        payload = {
            "guardrailId": config.guardrail_id,
            "promptUserId": usrid,
            "prompt": text,
            "isInput": False,
            "guardPromptId": config.last_guard_prompt_id,
        }

        # Prepare headers
        headers = {
            "X-Public-Key": config.public_key,
            "X-Secret-Key": config.secrect_key,
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                endpoint, json=payload, headers=headers, timeout=120
            )
            response.raise_for_status()
            json = response.json()

            return json["data"]["result"]
        except requests.RequestException as error:
            return error


guard = Detect()


class DatasetClient:
    """Class for managing datasets in Tmam.

    Attributes:
        id (str): Unique identifier of the dataset.
        name (str): Name of the dataset.
        description (Optional[str]): Description of the dataset.
        metadata (Optional[typing.Any]): Additional metadata of the dataset.
        last_run_at (datetime): Timestamp of last run.
        created_at (datetime): Timestamp of dataset creation.
        updated_at (datetime): Timestamp of the last update to the dataset.
        items (List[DatasetItemClient]): List of dataset items associated with the dataset.
    """

    id: str
    name: str
    description: Optional[str]
    metadata: Optional[Any]
    last_run_at: dt.datetime
    created_at: dt.datetime
    updated_at: dt.datetime
    items: List

    def __init__(
        self,
        _tmam_dataset: "Dataset",
        dataset_data: DatasetModel,
        items: List,
    ):
        """Initialize the DatasetClient."""
        self.id = dataset_data.id
        self.name = dataset_data.name
        self.description = dataset_data.description
        self.metadata = dataset_data.metadata
        self.last_run_at = dataset_data.last_run_at
        self.created_at = dataset_data.created_at
        self.updated_at = dataset_data.updated_at
        self.items = items
        self._tmam: "Dataset" = _tmam_dataset

    def run_experiment(
        self,
        *,
        name: str,
        run_name: Optional[str] = None,
        description: Optional[str] = None,
        task: TaskFunction,
        evaluators: List[EvaluatorFunction] = [],
        run_evaluators: List[RunEvaluatorFunction] = [],
        max_concurrency: int = 50,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ExperimentResult:
        """Run an experiment on this Tmam dataset with automatic tracking.

        This is a convenience method that runs an experiment using all items in this
        dataset. It automatically creates a dataset run in Tmam for tracking and
        comparison purposes, linking all experiment results to the dataset.
        """
        return self._tmam.run_experiment(
            name=name,
            run_name=run_name,
            description=description,
            data=self.items,
            task=task,
            evaluators=evaluators,
            run_evaluators=run_evaluators,
            max_concurrency=max_concurrency,
            metadata=metadata,
        )


# class DatasetItemClient:
#     """Class for managing dataset items in Tmam.

#     Args:
#         id (str): Unique identifier of the dataset item.
#         status (DatasetStatus): The status of the dataset item. Can be either 'ACTIVE' or 'ARCHIVED'.
#         input (Any): Input data of the dataset item.
#         expected_output (Optional[Any]): Expected output of the dataset item.
#         metadata (Optional[Any]): Additional metadata of the dataset item.
#         source_trace_id (Optional[str]): Identifier of the source trace.
#         source_observation_id (Optional[str]): Identifier of the source observation.
#         dataset_id (str): Identifier of the dataset to which this item belongs.
#         dataset_name (str): Name of the dataset to which this item belongs.
#         created_at (datetime): Timestamp of dataset item creation.
#         updated_at (datetime): Timestamp of the last update to the dataset item.
#         tmam (Tmam): Instance of Tmam client for API interactions.

#     Example:
#         ```python
#         from tmam import Tmam

#         tmam = Tmam()

#         dataset = tmam.get_dataset("<dataset_name>")

#         for item in dataset.items:
#             # Generate a completion using the input of every item
#             completion, generation = llm_app.run(item.input)

#             # Evaluate the completion
#             generation.score(
#                 name="example-score",
#                 value=1
#             )
#         ```
#     """

#     log = logging.getLogger("tmam")

#     id: str
#     status: Any
#     input: Any
#     expected_output: Optional[Any]
#     metadata: Optional[Any]
#     source_trace_id: Optional[str]
#     source_observation_id: Optional[str]
#     dataset_id: str
#     dataset_name: str
#     created_at: dt.datetime
#     updated_at: dt.datetime

#     tmam: "Tmam"

#     def __init__(self, dataset_item: Any, tmam: "Tmam"):
#         """Initialize the DatasetItemClient."""
#         self.id = dataset_item.id
#         self.status = dataset_item.status
#         self.input = dataset_item.input
#         self.expected_output = dataset_item.expected_output
#         self.metadata = dataset_item.metadata
#         self.source_trace_id = dataset_item.source_trace_id
#         self.source_observation_id = dataset_item.source_observation_id
#         self.dataset_id = dataset_item.dataset_id
#         self.dataset_name = dataset_item.dataset_name
#         self.created_at = dataset_item.created_at
#         self.updated_at = dataset_item.updated_at

#         self.tmam = tmam

#     @_agnosticcontextmanager
#     def run(
#         self,
#         *,
#         run_name: str,
#         run_metadata: Optional[Any] = None,
#         run_description: Optional[str] = None,
#     ) -> Generator[TmamSpan, None, None]:
#         """Create a context manager for the dataset item run that links the execution to a Tmam trace.

#         This method is a context manager that creates a trace for the dataset run and yields a span
#         that can be used to track the execution of the run.

#         Args:
#             run_name (str): The name of the dataset run.
#             run_metadata (Optional[Any]): Additional metadata to include in dataset run.
#             run_description (Optional[str]): Description of the dataset run.

#         Yields:
#             span: A TmamSpan that can be used to trace the execution of the run.
#         """
#         trace_name = f"Dataset run: {run_name}"

#         with self.tmam.start_as_current_span(name=trace_name) as span:
#             span.update_trace(
#                 name=trace_name,
#                 metadata={
#                     "dataset_item_id": self.id,
#                     "run_name": run_name,
#                     "dataset_id": self.dataset_id,
#                 },
#             )

#             self.log.debug(
#                 f"Creating dataset run item: run_name={run_name} id={self.id} trace_id={span.trace_id}"
#             )

#             self.tmam.api.dataset_run_items.create(
#                 request=CreateDatasetRunItemRequest(
#                     runName=run_name,
#                     datasetItemId=self.id,
#                     traceId=span.trace_id,
#                     metadata=run_metadata,
#                     runDescription=run_description,
#                 )
#             )

#             yield span


# class Dataset:
#     """
#         Represents a collection of input-output pairs used for evaluation, benchmarking,
#     or fine-tuning of language models. Supports storing, managing, and retrieving
#     dataset entries for consistent experimentation and comparison.

#     Attributes:
#         create_dataset (str): The data associated with the dataset entry.
#     """

#     def create_dataset(
#         self,
#         name: str,
#         description: str | None = None,
#         metadata: dict[str, str] | None = None,
#     ) -> bool:
#         """
#             Represents a collection of input-output pairs used for evaluation, benchmarking,
#         or fine-tuning of language models. Supports storing, managing, and retrieving
#         dataset entries for consistent experimentation and comparison.

#         Args:
#             name (str): dataset name.
#             description Optional[str]: description of the dataset.
#             metadata (Optional[dict[str, str]]): metadata of the dataset.
#         """

#         config = TmamConfig()

#         if (
#             config.url is None
#             or config.public_key is None
#             or config.secrect_key is None
#         ):
#             raise ValueError("make sure tmam.init is defined")

#         endpoint = config.url + "/dataset"

#         payload = {"name": name, "description": description, "metaData": metadata}

#         # Prepare headers
#         headers = {
#             "X-Public-Key": config.public_key,
#             "X-Secret-Key": config.secrect_key,
#             "Content-Type": "application/json",
#         }

#         try:
#             response = requests.post(
#                 endpoint, json=payload, headers=headers, timeout=120
#             )
#             response.raise_for_status()
#             json = response.json()

#             return json["data"]["result"]
#         except requests.RequestException as error:
#             return False

#     def create_dataset_item(
#         self,
#         dataset_name: str,
#         input: dict[str, str] | None = None,
#         expected_output: dict[str, str] | None = None,
#         metadata: dict[str, str] | None = None,
#     ) -> bool:
#         """
#             Represents a collection of input-output pairs used for evaluation, benchmarking,
#         or fine-tuning of language models. Supports storing, managing, and retrieving
#         dataset entries for consistent experimentation and comparison.

#         Args:
#             dataset_name (str): dataset name.
#             input (Optional[dict[str, str]]): input of the dataset.
#             expected_output (Optional[dict[str, str]]): expected_output of the dataset.
#             metadata (Optional[dict[str, str]]): metadata of the dataset.
#         """

#         config = TmamConfig()

#         if (
#             config.url is None
#             or config.public_key is None
#             or config.secrect_key is None
#         ):
#             raise ValueError("make sure tmam.init is defined")

#         endpoint = config.url + "/dataset/item"

#         payload = {
#             "dataset_name": dataset_name,
#             "input": input,
#             "expected_output": expected_output,
#             "metaData": metadata,
#         }

#         # Prepare headers
#         headers = {
#             "X-Public-Key": config.public_key,
#             "X-Secret-Key": config.secrect_key,
#             "Content-Type": "application/json",
#         }

#         try:
#             response = requests.post(
#                 endpoint, json=payload, headers=headers, timeout=120
#             )
#             response.raise_for_status()
#             json = response.json()

#             return json["data"]["result"]
#         except requests.RequestException as error:
#             return False

#     def get_dataset(
#         self,
#         name: str,
#     ) -> "DatasetClient":
#         """
#         Retrieve and returns the result from Tmam Dataset.

#         Args:
#             name (str): dataset name.
#         """

#         config = TmamConfig()

#         if (
#             config.url is None
#             or config.public_key is None
#             or config.secrect_key is None
#         ):
#             raise ValueError("make sure tmam.init is defined")

#         endpoint = config.url + "/dataset/items"

#         payload = {
#             "name": name,
#         }

#         # Prepare headers
#         headers = {
#             "X-Public-Key": config.public_key,
#             "X-Secret-Key": config.secrect_key,
#             "Content-Type": "application/json",
#         }

#         try:
#             response = requests.post(
#                 endpoint, json=payload, headers=headers, timeout=120
#             )
#             response.raise_for_status()
#             json = response.json()

#             # return json["data"]["result"]
#             dataset = json["data"]
#             dataset_items = json["data"]["items"]

#             items = [DatasetItemClient(i, tmam=self) for i in dataset_items]

#             return DatasetClient(self, dataset, items=items)
#         except requests.RequestException as error:
#             return error

#     def run_experiment(
#         self,
#         *,
#         name: str,
#         run_name: Optional[str] = None,
#         description: Optional[str] = None,
#         data: ExperimentData,
#         task: TaskFunction,
#         evaluators: List[EvaluatorFunction] = [],
#         run_evaluators: List[RunEvaluatorFunction] = [],
#         max_concurrency: int = 50,
#         metadata: Optional[Dict[str, str]] = None,
#     ) -> ExperimentResult:
#         return cast(
#             ExperimentResult,
#             run_async_safely(
#                 self._run_experiment_async(
#                     name=name,
#                     run_name=self._create_experiment_run_name(
#                         name=name, run_name=run_name
#                     ),
#                     description=description,
#                     data=data,
#                     task=task,
#                     evaluators=evaluators or [],
#                     run_evaluators=run_evaluators or [],
#                     max_concurrency=max_concurrency,
#                     metadata=metadata,
#                 ),
#             ),
#         )

#     async def _run_experiment_async(
#         self,
#         *,
#         name: str,
#         run_name: str,
#         description: Optional[str],
#         data: ExperimentData,
#         task: TaskFunction,
#         evaluators: List[EvaluatorFunction],
#         run_evaluators: List[RunEvaluatorFunction],
#         max_concurrency: int,
#         metadata: Optional[Dict[str, Any]] = None,
#     ) -> ExperimentResult:

#         # Set up concurrency control
#         semaphore = asyncio.Semaphore(max_concurrency)

#         # Process all items
#         async def process_item(item: ExperimentItem) -> ExperimentItemResult:
#             async with semaphore:
#                 return await self._process_experiment_item(
#                     item, task, evaluators, name, run_name, description, metadata
#                 )

#         # Run all items concurrently
#         tasks = [process_item(item) for item in data]
#         item_results = await asyncio.gather(*tasks, return_exceptions=True)

#         # Filter out any exceptions and log errors
#         valid_results: List[ExperimentItemResult] = []
#         for i, result in enumerate(item_results):
#             if isinstance(result, Exception):
#                 pass
#             elif isinstance(result, ExperimentItemResult):
#                 valid_results.append(result)  # type: ignore

#         # Run experiment-level evaluators
#         run_evaluations: List[Evaluation] = []
#         for run_evaluator in run_evaluators:
#             try:
#                 evaluations = await run_evaluator_def(
#                     run_evaluator, item_results=valid_results
#                 )
#                 run_evaluations.extend(evaluations)
#             except Exception as e:
#                 pass

#         # Generate dataset run URL if applicable
#         dataset_run_id = valid_results[0].dataset_run_id if valid_results else None
#         dataset_run_url = None
#         if dataset_run_id and data:
#             try:
#                 # Check if the first item has dataset_id (for DatasetItem objects)
#                 first_item = data[0]
#                 dataset_id = None

#                 if hasattr(first_item, "dataset_id"):
#                     dataset_id = getattr(first_item, "dataset_id", None)

#                 if dataset_id:
#                     project_id = self._get_project_id()

#                     if project_id:
#                         dataset_run_url = f"{self._base_url}/project/{project_id}/datasets/{dataset_id}/runs/{dataset_run_id}"

#             except Exception:
#                 pass  # URL generation is optional

#         # Store run-level evaluations as scores
#         for evaluation in run_evaluations:
#             try:
#                 if dataset_run_id:
#                     self.create_score(
#                         dataset_run_id=dataset_run_id,
#                         name=evaluation.name or "<unknown>",
#                         value=evaluation.value,  # type: ignore
#                         comment=evaluation.comment,
#                         metadata=evaluation.metadata,
#                         data_type=evaluation.data_type,  # type: ignore
#                         config_id=evaluation.config_id,
#                     )

#             except Exception as e:
#                 pass

#         # Flush scores and traces
#         self.flush()

#         return ExperimentResult(
#             name=name,
#             run_name=run_name,
#             description=description,
#             item_results=valid_results,
#             run_evaluations=run_evaluations,
#             dataset_run_id=dataset_run_id,
#             dataset_run_url=dataset_run_url,
#         )

#     async def _process_experiment_item(
#         self,
#         item: ExperimentItem,
#         task: Callable,
#         evaluators: List[Callable],
#         experiment_name: str,
#         experiment_run_name: str,
#         experiment_description: Optional[str],
#         experiment_metadata: Optional[Dict[str, Any]] = None,
#     ) -> ExperimentItemResult:
#         span_name = "experiment-item-run"

#         with self.start_as_current_span(name=span_name) as span:
#             try:
#                 input_data = (
#                     item.get("input")
#                     if isinstance(item, dict)
#                     else getattr(item, "input", None)
#                 )

#                 if input_data is None:
#                     raise ValueError("Experiment Item is missing input. Skipping item.")

#                 expected_output = (
#                     item.get("expected_output")
#                     if isinstance(item, dict)
#                     else getattr(item, "expected_output", None)
#                 )

#                 item_metadata = (
#                     item.get("metadata")
#                     if isinstance(item, dict)
#                     else getattr(item, "metadata", None)
#                 )

#                 final_observation_metadata = {
#                     "experiment_name": experiment_name,
#                     "experiment_run_name": experiment_run_name,
#                     **(experiment_metadata or {}),
#                 }

#                 trace_id = span.trace_id
#                 dataset_id = None
#                 dataset_item_id = None
#                 dataset_run_id = None

#                 # Link to dataset run if this is a dataset item
#                 if hasattr(item, "id") and hasattr(item, "dataset_id"):
#                     try:
#                         # Use sync API to avoid event loop issues when run_async_safely
#                         # creates multiple event loops across different threads
#                         dataset_run_item = await asyncio.to_thread(
#                             self.api.dataset_run_items.create,
#                             request=CreateDatasetRunItemRequest(
#                                 runName=experiment_run_name,
#                                 runDescription=experiment_description,
#                                 metadata=experiment_metadata,
#                                 datasetItemId=item.id,  # type: ignore
#                                 traceId=trace_id,
#                                 observationId=span.id,
#                             ),
#                         )

#                         dataset_run_id = dataset_run_item.dataset_run_id

#                     except Exception as e:
#                         pass

#                 if (
#                     not isinstance(item, dict)
#                     and hasattr(item, "dataset_id")
#                     and hasattr(item, "id")
#                 ):
#                     dataset_id = item.dataset_id
#                     dataset_item_id = item.id

#                     final_observation_metadata.update(
#                         {"dataset_id": dataset_id, "dataset_item_id": dataset_item_id}
#                     )

#                 if isinstance(item_metadata, dict):
#                     final_observation_metadata.update(item_metadata)

#                 experiment_id = dataset_run_id or self._create_observation_id()
#                 experiment_item_id = (
#                     dataset_item_id or get_sha256_hash_hex(_serialize(input_data))[:16]
#                 )
#                 span._otel_span.set_attributes(
#                     {
#                         k: v
#                         for k, v in {
#                             TmamOtelSpanAttributes.ENVIRONMENT: TMAM_SDK_EXPERIMENT_ENVIRONMENT,
#                             TmamOtelSpanAttributes.EXPERIMENT_DESCRIPTION: experiment_description,
#                             TmamOtelSpanAttributes.EXPERIMENT_ITEM_EXPECTED_OUTPUT: _serialize(
#                                 expected_output
#                             ),
#                         }.items()
#                         if v is not None
#                     }
#                 )

#                 with _propagate_attributes(
#                     experiment=PropagatedExperimentAttributes(
#                         experiment_id=experiment_id,
#                         experiment_name=experiment_run_name,
#                         experiment_metadata=_serialize(experiment_metadata),
#                         experiment_dataset_id=dataset_id,
#                         experiment_item_id=experiment_item_id,
#                         experiment_item_metadata=_serialize(item_metadata),
#                         experiment_item_root_observation_id=span.id,
#                     )
#                 ):
#                     output = await self._run_task(task, item)

#                 span.update(
#                     input=input_data,
#                     output=output,
#                     metadata=final_observation_metadata,
#                 )

#                 # Run evaluators
#                 evaluations = []

#                 for evaluator in evaluators:
#                     try:
#                         eval_metadata: Optional[Dict[str, Any]] = None

#                         if isinstance(item, dict):
#                             eval_metadata = item.get("metadata")
#                         elif hasattr(item, "metadata"):
#                             eval_metadata = item.metadata

#                         eval_results = await self._run_evaluator(
#                             evaluator,
#                             input=input_data,
#                             output=output,
#                             expected_output=expected_output,
#                             metadata=eval_metadata,
#                         )
#                         evaluations.extend(eval_results)

#                         # Store evaluations as scores
#                         for evaluation in eval_results:
#                             self.create_score(
#                                 trace_id=trace_id,
#                                 observation_id=span.id,
#                                 name=evaluation.name,
#                                 value=evaluation.value,  # type: ignore
#                                 comment=evaluation.comment,
#                                 metadata=evaluation.metadata,
#                                 config_id=evaluation.config_id,
#                                 data_type=evaluation.data_type,  # type: ignore
#                             )

#                     except Exception as e:
#                         pass

#                 return ExperimentItemResult(
#                     item=item,
#                     output=output,
#                     evaluations=evaluations,
#                     trace_id=trace_id,
#                     dataset_run_id=dataset_run_id,
#                 )

#             except Exception as e:
#                 span.update(
#                     output=f"Error: {str(e)}", level="ERROR", status_message=str(e)
#                 )
#                 raise e

#     def _create_experiment_run_name(
#         self, *, name: Optional[str] = None, run_name: Optional[str] = None
#     ) -> str:
#         if run_name:
#             return run_name

#         iso_timestamp = _get_timestamp().isoformat().replace("+00:00", "Z")

#         return f"{name} - {iso_timestamp}"

#     async def _run_task(task: TaskFunction, item: ExperimentItem) -> Any:
#         """Run a task function and handle sync/async."""
#         result = task(item=item)

#         # Handle async tasks
#         if asyncio.iscoroutine(result):
#             result = await result

#         return result

#     async def _run_evaluator(
#         evaluator: Union[EvaluatorFunction, RunEvaluatorFunction], **kwargs: Any
#     ) -> List[Evaluation]:
#         """Run an evaluator function and normalize the result."""
#         try:
#             result = evaluator(**kwargs)

#             # Handle async evaluators
#             if asyncio.iscoroutine(result):
#                 result = await result

#             # Normalize to list
#             if isinstance(result, (dict, Evaluation)):
#                 return [result]  # type: ignore

#             elif isinstance(result, list):
#                 return result

#             else:
#                 return []

#         except Exception as e:
#             evaluator_name = getattr(evaluator, "__name__", "unknown_evaluator")
#             logging.getLogger("tmam").error(f"Evaluator {evaluator_name} failed: {e}")
#             return []
