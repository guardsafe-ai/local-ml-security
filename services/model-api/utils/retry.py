"""
Retry utilities with exponential backoff
Provides retry decorators for database, HTTP, and MLflow operations
"""

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)
import logging
import asyncpg
import httpx
import mlflow.exceptions

logger = logging.getLogger(__name__)

# Database operations retry
db_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((
        asyncpg.PostgresError,
        asyncpg.InterfaceError,
        asyncpg.ConnectionDoesNotExistError
    )),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)

# HTTP operations retry
http_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((
        httpx.RequestError,
        httpx.HTTPStatusError,
        httpx.TimeoutException,
        httpx.ConnectError
    )),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)

# MLflow operations retry
mlflow_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    retry=retry_if_exception_type((
        mlflow.exceptions.MlflowException,
        mlflow.exceptions.RestException
    )),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)

# Redis operations retry
redis_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=5),
    retry=retry_if_exception_type((
        ConnectionError,
        TimeoutError
    )),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
