"""
Setups up OpenTelemetry tracer
"""

import os
from opentelemetry import trace
from opentelemetry.sdk.resources import (
    SERVICE_NAME,
    TELEMETRY_SDK_NAME,
    DEPLOYMENT_ENVIRONMENT,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter


# Global flag to check if the tracer provider initialization is complete.
TRACER_SET = False


def setup_tracing(
    application_name, environment, tracer, url, public_key, secrect_key, disable_batch
):
    """
    Sets up tracing with OpenTelemetry.
    Initializes the tracer provider and configures the span processor and exporter.
    """

    # If an external tracer is provided, return it immediately.
    if tracer is not None:
        return tracer

    # Proceed with setting up a new tracer or configuration only if TRACER_SET is False.
    # pylint: disable=global-statement
    global TRACER_SET

    try:
        # Disable Haystack Auto Tracing
        os.environ["HAYSTACK_AUTO_TRACE_ENABLED"] = "false"

        if not TRACER_SET:
            # Create a resource with the service name attribute.
            resource = Resource.create(
                attributes={
                    SERVICE_NAME: application_name,
                    DEPLOYMENT_ENVIRONMENT: environment,
                    TELEMETRY_SDK_NAME: "tmam",
                }
            )

            # Initialize the TracerProvider with the created resource.
            trace.set_tracer_provider(TracerProvider(resource=resource))

            # Only set environment variables if you have a non-None value.
            if url is not None:
                os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = url

            if public_key is not None and secrect_key is not None:
                headers = {
                    "X-Public-Key": public_key,
                    "X-Secret-Key": secrect_key
                }
                headers_str = ",".join(
                    f"{key}={value}"
                    for key, value in headers.items()
                    if value is not None
                )
                os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = headers_str

            # Configure the span exporter and processor based on whether the endpoint is effectively set.
            span_exporter = OTLPSpanExporter()
            # pylint: disable=line-too-long
            span_processor = (
                BatchSpanProcessor(span_exporter)
                if not disable_batch
                else SimpleSpanProcessor(span_exporter)
            )

            trace.get_tracer_provider().add_span_processor(span_processor)

            TRACER_SET = True

        return trace.get_tracer(__name__)

    # pylint: disable=bare-except
    except:
        return None
