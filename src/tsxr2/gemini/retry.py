"""Retry logic with exponential backoff for API calls.

Provides decorators for handling transient API failures
with configurable retry behavior.
"""

import logging
import random
import time
from functools import wraps
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for retry with exponential backoff.

    Retries the decorated function on exception, with exponentially
    increasing delays between retries.

    Args:
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay in seconds.
        max_delay: Maximum delay cap in seconds.
        exponential_base: Base for exponential growth.
        jitter: Whether to add random jitter to delays.

    Returns:
        Decorated function with retry behavior.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Exception | None = None

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        # Calculate delay with exponential backoff
                        delay = min(
                            base_delay * (exponential_base**attempt),
                            max_delay,
                        )

                        # Add jitter (±25% of delay)
                        if jitter:
                            delay = delay * (0.75 + random.random() * 0.5)

                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries} failed: {e}. "
                            f"Retrying in {delay:.2f}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"All {max_retries} attempts failed. Last error: {e}"
                        )

            # Re-raise the last exception after all retries exhausted
            if last_exception:
                raise last_exception
            raise RuntimeError("Unexpected state: no exception captured")

        return wrapper

    return decorator
