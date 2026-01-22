from __future__ import annotations

import logging
import os
from typing import Dict, Optional


_OTEL_INITIALIZED = False
_OTEL_INSTRUMENTED = False


def _parse_headers(raw: Optional[str]) -> Dict[str, str]:
    if not raw:
        return {}
    headers: Dict[str, str] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item or "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key and value:
            headers[key] = value
    return headers


def _parse_resource_attributes(raw: Optional[str]) -> Dict[str, str]:
    if not raw:
        return {}
    attrs: Dict[str, str] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item or "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key and value:
            attrs[key] = value
    return attrs


def _resolve_http_endpoint(
    base: Optional[str], signal: str, override: Optional[str]
) -> Optional[str]:
    endpoint = (override or base or "").strip()
    if not endpoint:
        return None
    if "/v1/" in endpoint:
        return endpoint
    return endpoint.rstrip("/") + f"/v1/{signal}"


def _should_enable_exporter(name: Optional[str]) -> bool:
    if not name:
        return True
    return name.strip().lower() not in {"none", "off", "false", "0"}


def init_otel(service_name: Optional[str] = None) -> bool:
    global _OTEL_INITIALIZED
    if _OTEL_INITIALIZED:
        return True
    base_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    traces_endpoint_env = os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
    logs_endpoint_env = os.getenv("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT")
    protocol = (os.getenv("OTEL_EXPORTER_OTLP_PROTOCOL") or "grpc").strip().lower()
    use_http = protocol.startswith("http")
    headers = _parse_headers(os.getenv("OTEL_EXPORTER_OTLP_HEADERS"))
    trace_exporter = os.getenv("OTEL_TRACES_EXPORTER", "otlp")
    log_exporter = os.getenv("OTEL_LOGS_EXPORTER", "otlp")
    enable_traces = _should_enable_exporter(trace_exporter)
    enable_logs = _should_enable_exporter(log_exporter)
    if not (enable_traces or enable_logs):
        return False
    if not (base_endpoint or traces_endpoint_env or logs_endpoint_env):
        return False

    try:
        from opentelemetry import trace
        from opentelemetry._logs import set_logger_provider
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
        from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
    except Exception:
        return False

    service = service_name or os.getenv("OTEL_SERVICE_NAME") or "nlp-crop-calendar"
    resource_attrs = _parse_resource_attributes(
        os.getenv("OTEL_RESOURCE_ATTRIBUTES")
    )
    resource = Resource.create({"service.name": service, **resource_attrs})

    configured = False
    if enable_traces:
        traces_endpoint = (
            _resolve_http_endpoint(base_endpoint, "traces", traces_endpoint_env)
            if use_http
            else (traces_endpoint_env or base_endpoint)
        )
        if traces_endpoint:
            try:
                if use_http:
                    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                        OTLPSpanExporter,
                    )
                else:
                    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                        OTLPSpanExporter,
                    )
            except Exception:
                traces_endpoint = None
        if traces_endpoint:
            tracer_provider = TracerProvider(resource=resource)
            span_exporter = OTLPSpanExporter(
                endpoint=traces_endpoint, headers=headers
            )
            tracer_provider.add_span_processor(
                BatchSpanProcessor(span_exporter)
            )
            trace.set_tracer_provider(tracer_provider)
            configured = True

    if enable_logs:
        logs_endpoint = (
            _resolve_http_endpoint(base_endpoint, "logs", logs_endpoint_env)
            if use_http
            else (logs_endpoint_env or base_endpoint)
        )
        if logs_endpoint:
            try:
                if use_http:
                    from opentelemetry.exporter.otlp.proto.http._log_exporter import (
                        OTLPLogExporter,
                    )
                else:
                    from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
                        OTLPLogExporter,
                    )
            except Exception:
                logs_endpoint = None
        if logs_endpoint:
            logger_provider = LoggerProvider(resource=resource)
            log_exporter_instance = OTLPLogExporter(
                endpoint=logs_endpoint, headers=headers
            )
            logger_provider.add_log_record_processor(
                BatchLogRecordProcessor(log_exporter_instance)
            )
            set_logger_provider(logger_provider)
            handler = LoggingHandler(
                level=logging.INFO, logger_provider=logger_provider
            )
            logging.getLogger().addHandler(handler)
            configured = True

    if not configured:
        return False
    _OTEL_INITIALIZED = True
    return True


def instrument_fastapi(app: object) -> bool:
    global _OTEL_INSTRUMENTED
    if _OTEL_INSTRUMENTED:
        return True
    try:
        from opentelemetry.instrumentation.fastapi import (
            FastAPIInstrumentor,
        )
    except Exception:
        return False
    try:
        FastAPIInstrumentor.instrument_app(app)
    except Exception:
        return False
    _OTEL_INSTRUMENTED = True
    return True


def instrument_httpx() -> bool:
    try:
        from opentelemetry.instrumentation.httpx import (
            HTTPXClientInstrumentor,
        )
    except Exception:
        return False
    try:
        HTTPXClientInstrumentor().instrument()
    except Exception:
        return False
    return True
