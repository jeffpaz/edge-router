"""Per-request "private" flag and the cloud guard.

A private request must never reach a cloud provider. The routing layer already
builds a lane list without cloud clients; this is the second, independent guard:
every cloud call path calls assert_cloud_allowed() before touching the network,
and it raises while the current request is private.

No imports from the rest of the app, so any module can use it.
"""
from contextlib import contextmanager
from contextvars import ContextVar

_private: ContextVar[bool] = ContextVar("private_request", default=False)


class PrivateLeakError(RuntimeError):
    """A cloud call was attempted while handling a private request."""


def is_private() -> bool:
    return _private.get()


@contextmanager
def private_scope():
    """Mark the enclosed work (and tasks created inside it) as a private request."""
    token = _private.set(True)
    try:
        yield
    finally:
        _private.reset(token)


def assert_cloud_allowed(provider: str) -> None:
    if _private.get():
        raise PrivateLeakError(f"cloud call to {provider!r} blocked: request is private")
