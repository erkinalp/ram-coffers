"""Node-attributed failures shared by the transports and the dispatcher.

Every failure names the console it came from, so a coordinator log says *which*
of ~82.5k nodes broke rather than "the cluster is down" — the coordinator-side
equivalent of DaCS's remote error handler, which delivers accelerator errors
back to the host process (DaCS Programmer's Guide, SDK 3.0).

The ``safe_to_retry`` class attribute encodes whether the request can be re-sent
**without any risk of the expert running twice**:

``True``  the request provably never reached the node (no endpoint, connect
          refused, write failed before the frame was accepted), so retrying is
          *at-most-once*: the expert is applied exactly once or not at all.
``False`` the node may already be computing (or may have computed) the result;
          re-sending would make the call *at-least-once*. The dispatcher only
          does this when explicitly opted into via :class:`RetryPolicy`.
"""

from __future__ import annotations


class TransportError(RuntimeError):
    """Base class for node-attributed transport failures."""

    #: See module docstring.
    safe_to_retry = False

    def __init__(self, node_id: str, message: str):
        super().__init__(f"[{node_id}] {message}")
        self.node_id = node_id


class NodeConnectError(TransportError):
    """The node could not be reached; the request never left the coordinator."""

    safe_to_retry = True


class NodeError(TransportError):
    """The node answered with a P3XC ``ERR`` frame.

    A definitive negative answer: the node produced no output activation, so a
    retry elsewhere cannot double-count an expert contribution. It is still not
    ``safe_to_retry`` by default, because an ``ERR`` usually means a mismatched
    or broken expert and retrying the same node just fails again; enable it per
    call site with ``RetryPolicy(retry_on_node_error=True)``.
    """

    def __init__(self, node_id: str, layer: int, expert: int, token_id: int):
        super().__init__(node_id,
                         f"expert node returned ERR for layer={layer} "
                         f"expert={expert} token={token_id}")
        self.layer = layer
        self.expert = expert
        self.token_id = token_id


class PoolExhausted(TransportError):
    """Every pooled connection already has this correlation key in flight.

    The request was never written, so it is safe to retry once a slot frees.
    """

    safe_to_retry = True


class NodeTimeout(TransportError):
    """No response within the deadline. The request may still be executing."""


class NodeDisconnected(TransportError):
    """The connection dropped while requests were in flight."""


class TransportClosed(TransportError):
    """The transport (or one of its connections) was closed locally."""
