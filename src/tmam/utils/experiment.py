from __future__ import annotations

import asyncio
import threading
from typing import Any, Coroutine, List, Union

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tmam.model.dataset import Evaluation, EvaluatorFunction, RunEvaluatorFunction


def format_value(value: Any) -> str:
    """Format a value for display."""
    if isinstance(value, str):
        return value[:50] + "..." if len(value) > 50 else value
    return str(value)


def run_async_safely(coro: Coroutine[Any, Any, Any]) -> Any:
    """Safely run an async coroutine, handling existing event loops.

    This function detects if there's already a running event loop and uses
    a separate thread if needed to avoid the "asyncio.run() cannot be called
    from a running event loop" error. This is particularly useful in environments
    like Jupyter notebooks, FastAPI applications, or other async frameworks.

    Args:
        coro: The coroutine to run

    Returns:
        The result of the coroutine

    Raises:
        Any exception raised by the coroutine

    Example:
        ```python
        # Works in both sync and async contexts
        async def my_async_function():
            await asyncio.sleep(1)
            return "done"

        result = run_async_safely(my_async_function())
        ```
    """
    try:
        # Check if there's already a running event loop
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop, safe to use asyncio.run()
        return asyncio.run(coro)

    if loop and loop.is_running():
        # There's a running loop, use a separate thread
        thread = _RunAsyncThread(coro)
        thread.start()
        thread.join()

        if thread.exception:
            raise thread.exception
        return thread.result
    else:
        # Loop exists but not running, safe to use asyncio.run()
        return asyncio.run(coro)


class _RunAsyncThread(threading.Thread):
    """Helper thread class for running async coroutines in a separate thread."""

    def __init__(self, coro: Coroutine[Any, Any, Any]) -> None:
        self.coro = coro
        self.result: Any = None
        self.exception: Exception | None = None
        super().__init__()

    def run(self) -> None:
        try:
            self.result = asyncio.run(self.coro)
        except Exception as e:
            self.exception = e


async def run_evaluator_def(
    evaluator: Union[EvaluatorFunction, RunEvaluatorFunction], **kwargs: Any
) -> List[Evaluation]:
    """Run an evaluator function and normalize the result."""
    try:
        result = evaluator(**kwargs)

        # Handle async evaluators
        if asyncio.iscoroutine(result):
            result = await result

        # Normalize to list
        if isinstance(result, (dict, Evaluation)):
            return [result]  # type: ignore

        elif isinstance(result, list):
            return result

        else:
            return []

    except Exception as e:
        return []
