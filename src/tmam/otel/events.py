"""
Setups up OpenTelemetry events emitter
"""

import os
from opentelemetry import _events, _logs
from opentelemetry.sdk.resources import SERVICE_NAME, TELEMETRY_SDK_NAME, DEPLOYMENT_ENVIRONMENT
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk._events import EventLoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor, SimpleLogRecordProcessor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter

# Global flag to check if the events provider initialization is complete.
EVENTS_SET = False

def setup_events(application_name, environment, event_logger, url, public_key, secrect_key, disable_batch):
    """Setup OpenTelemetry events with the given configuration.

    Args:
        application_name: Name of the application
        environment: Deployment environment
        event_logger: Optional pre-configured event logger provider
        otlp_endpoint: Optional OTLP endpoint for exporter
        otlp_headers: Optional headers for OTLP exporter

    Returns:
        EventLoggerProvider: The configured event logger provider
    """
    # If an external events_logger is provided, return it immediately.
    if event_logger:
        return event_logger

    # Proceed with setting up a new events or configuration only if EVENTS_SET is False.
    global EVENTS_SET

    try:
        if not EVENTS_SET:
            # Create resource with service and environment information
            resource = Resource.create(attributes={
                SERVICE_NAME: application_name,
                DEPLOYMENT_ENVIRONMENT: environment,
                TELEMETRY_SDK_NAME: "tmam"}
            )

            # Initialize the LoggerProvider with the created resource.
            logger_provider = LoggerProvider(resource=resource)

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
            event_exporter = OTLPLogExporter()
            # pylint: disable=line-too-long
            logger_provider.add_log_record_processor(SimpleLogRecordProcessor(event_exporter)) if disable_batch else logger_provider.add_log_record_processor(BatchLogRecordProcessor(event_exporter))

            _logs.set_logger_provider(logger_provider)
            event_provider = EventLoggerProvider()
            _events.set_event_logger_provider(event_provider)

            EVENTS_SET = True

        return _events.get_event_logger(__name__)

    # pylint: disable=bare-except
    except:
        return None
