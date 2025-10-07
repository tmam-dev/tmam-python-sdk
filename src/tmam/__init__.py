# pylint: disable=broad-exception-caught
"""
The __init__.py module for the Tmam package.
This module sets up the Tmam configuration and instrumentation for various
large language models (LLMs).
"""

from typing import Dict, Optional
import logging
import os
from importlib.util import find_spec
from functools import wraps
from contextlib import contextmanager
import requests


# Import internal modules for setting up tracing and fetching pricing info.
from opentelemetry import trace as t
from opentelemetry.trace import SpanKind, Status, StatusCode, Span
from opentelemetry.sdk.resources import SERVICE_NAME, DEPLOYMENT_ENVIRONMENT
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
    version=None,
    should_compile=None,
    variables=None,
    meta_properties=None,
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
        "shouldCompile": should_compile,
        "variables": variables,
        "metaProperties": meta_properties,
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


def trace(wrapped):
    """
    Generates a telemetry wrapper for messages to collect metrics.
    """
    if not callable(wrapped):
        raise TypeError(
            f"@trace can only be applied to callable objects, got {type(wrapped).__name__}"
        )

    try:
        __trace = t.get_tracer_provider()
        tracer = __trace.get_tracer(__name__)
    except Exception as tracer_exception:
        logging.error(
            "Failed to initialize tracer: %s", tracer_exception, exc_info=True
        )
        raise

    @wraps(wrapped)
    def wrapper(*args, **kwargs):
        with tracer.start_as_current_span(
            name=wrapped.__name__,
            kind=SpanKind.CLIENT,
        ) as span:
            response = None
            try:
                response = wrapped(*args, **kwargs)
                span.set_attribute(
                    SemanticConvetion.GEN_AI_CONTENT_COMPLETION, response or ""
                )
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                span.record_exception(e)
                span.set_status(status=Status(StatusCode.ERROR), description=str(e))
                logging.error("Error in %s: %s", wrapped.__name__, e, exc_info=True)
                raise

            try:
                span.set_attribute("function.args", str(args))
                span.set_attribute("function.kwargs", str(kwargs))
                span.set_attribute(
                    SERVICE_NAME,
                    TmamConfig.application_name,
                )
                span.set_attribute(DEPLOYMENT_ENVIRONMENT, TmamConfig.environment)
            except Exception as meta_exception:
                logging.error(
                    "Failed to set metadata for %s: %s",
                    wrapped.__name__,
                    meta_exception,
                    exc_info=True,
                )

            return response

    return wrapper


class TracedSpan:
    """
    A wrapper class for an OpenTelemetry span that provides helper methods
    for setting result and metadata attributes on the span.

    Attributes:
        _span (Span): The underlying OpenTelemetry span.
    """

    def __init__(self, span):
        """
        Initializes the TracedSpan with the given span.

        Params:
            span (Span): The OpenTelemetry span to be wrapped.
        """

        self._span: Span = span

    def set_result(self, result):
        """
        Sets the result attribute on the underlying span.

        Params:
            result: The result to be set as an attribute on the span.
        """

        self._span.set_attribute(SemanticConvetion.GEN_AI_CONTENT_COMPLETION, result)

    def set_metadata(self, metadata: Dict):
        """
        Sets multiple attributes on the underlying span.

        Params:
            metadata (Dict): A dictionary of attributes to be set on the span.
        """

        self._span.set_attributes(attributes=metadata)

    def __enter__(self):
        """
        Enters the context of the TracedSpan, returning itself.

        Returns:
            TracedSpan: The instance of TracedSpan.
        """

        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        """
        Exits the context of the TracedSpan by ending the underlying span.
        """

        self._span.end()


@contextmanager
def start_trace(name: str):
    """
    A context manager that starts a new trace and provides a TracedSpan
    for usage within the context.

    Params:
        name (str): The name of the span.

    Yields:
        TracedSpan: The wrapped span for trace operations.
    """

    __trace = t.get_tracer_provider()
    with __trace.get_tracer(__name__).start_as_current_span(
        name,
        kind=SpanKind.CLIENT,
    ) as span:
        yield TracedSpan(span)


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
