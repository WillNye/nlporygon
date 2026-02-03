from __future__ import annotations

import logging
import os

import orjson
import structlog


def configure_logger(logger_name, log_level):
    if not structlog.is_configured():
        renderer = structlog.processors.JSONRenderer(serializer=orjson.dumps)
        structlog.configure(
            processors=[
                structlog.processors.add_log_level,
                structlog.contextvars.merge_contextvars,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper("%Y/%m/%d %H:%M:%S", utc=False),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.CallsiteParameterAdder(
                    parameters=[
                        structlog.processors.CallsiteParameter.FILENAME,
                        structlog.processors.CallsiteParameter.FUNC_NAME,
                        structlog.processors.CallsiteParameter.LINENO,
                    ]
                ),
                structlog.dev.set_exc_info,
                renderer,
            ],
            wrapper_class=structlog.make_filtering_bound_logger(
                logging.getLevelName(log_level)
            ),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=False,
        )

    return structlog.get_logger(logger_name)


logger = configure_logger("nlq_prompt", os.getenv("LOG_LEVEL", "INFO"))
