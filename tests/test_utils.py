"""Utility functions for testing."""

from typing import Any, AsyncGenerator, Type


async def create_async_error_generator(
    exception_class: Type[Exception], *args: Any, **kwargs: Any
) -> AsyncGenerator:
    """
    Create an async generator that raises an exception on the first call to __anext__.

    Args:
        exception_class: The exception class to raise
        *args: Positional arguments to pass to the exception constructor
        **kwargs: Keyword arguments to pass to the exception constructor

    Returns:
        An async generator function that raises the specified exception
    """
    # Raise the exception immediately
    raise exception_class(*args, **kwargs)
    # This line will never be reached, but it's here to satisfy the type checker
    yield  # type: ignore


def create_async_value_generator(*values):
    """
    Create an async generator that yields the provided values.

    Args:
        *values: Values to yield from the generator

    Returns:
        An async generator function that yields the provided values
    """

    async def value_generator(*generator_args, **generator_kwargs):
        for value in values:
            yield value

    return value_generator
