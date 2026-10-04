# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TurnLoop: drives an ``InteractiveAgent`` from its channels, one turn per wake.

The loop knows channels and turns, nothing else. It races the agent's
queue-mode channels, drains what is buffered, calls the turn method
(``handle`` by default) and reports what happened as events on the
agent's ``event_manager``:

- ``TurnBegan`` right before the turn method is called,
- ``TurnSettled`` once the turn has finished, however it finished,
- ``TurnLoopEnded`` when the loop can no longer wait for input.

Producers put on the channels directly; whatever records, displays or
persists turns subscribes to these events (and to the channels' own
``ChannelItemConsumed`` / ``ChannelItemsDiscarded``). Subscribers run
synchronously: ``TurnSettled`` subscribers have finished before the loop
races again and before ``cancel()`` returns. A subscriber that raises is
logged and does not stop the loop.

Events observe; they cannot stop a step. A host that must act before a
turn, and fail the turn if it cannot, passes ``before_turn`` to
:meth:`TurnLoop.start`.
"""

import asyncio
import contextvars
import logging
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import Annotated, Any, ClassVar, Literal

from pydantic import Field

from nooa.context_blocks import EventBase
from nooa.context_blocks.roles import Role

logger = logging.getLogger(__name__)

BeforeTurn = Callable[[dict[str, list[Any]]], Awaitable[None]]
"""Awaited with the notification before each turn; raising fails that turn."""

TurnKind = Literal["done", "need_input", "waiting", "cancelled", "error"]


class TurnCancelled(EventBase):
    """A person, a parent or the host stopped the turn before it finished.

    Appended to the agent's events when a cancel takes effect, after the
    interrupted cell's output, so the model sees at its next turn that it
    was stopped rather than that a cell failed. ``by`` says who stopped it
    (``"user"``, ``"parent:<name>"``, ``"host"``); ``interrupted`` is the
    tag of the interrupted cell's ``PythonOutput``, or ``None`` when the
    turn was stopped between cells (for example during a model call).
    """

    _role: ClassVar[Role] = Role.USER

    by: str
    interrupted: str | None = None


class TurnBegan(EventBase):
    """Published right before the loop calls the turn method.

    ``notification`` is what the turn receives (channel name → items).
    Every item in it has already been published as ``ChannelItemConsumed``.
    A runtime event: never recorded, never shown to the model.
    """

    _role: ClassVar[Role] = Role.RUNTIME_EVENT

    notification: Annotated[dict[str, list[Any]], Field(repr=False)] = Field(default_factory=dict)


class TurnSettled(EventBase):
    """Published once a turn has finished, however it finished.

    - ``done`` / ``need_input`` / ``waiting``: ``result`` is the turn's
      ``Done`` / ``NeedInput`` / ``Waiting``.
    - ``cancelled``: ``cancel(by=...)`` stopped it. ``cancelled_by`` names
      who; ``interrupted`` is the tag of the interrupted cell's output (or
      ``None``). ``TurnCancelled`` is already in the agent's events, unless
      ``ran`` is ``False`` (cancelled during ``before_turn``).
    - ``error``: ``message`` says what went wrong and ``error`` is the
      exception (``None`` for a turn that returned something other than a
      turn result).

    ``ran`` is ``False`` when the turn method was never called (``before_turn``
    raised, or a cancel came during it).

    A runtime event: never recorded, never shown to the model.
    """

    _role: ClassVar[Role] = Role.RUNTIME_EVENT

    kind: TurnKind
    result: Annotated[Any, Field(repr=False)] = None
    error: Annotated[Any, Field(repr=False)] = None
    message: str = ""
    cancelled_by: str | None = None
    interrupted: str | None = None
    ran: bool = True


class TurnLoopEnded(EventBase):
    """Published when the loop stops because it can no longer wait for input.

    ``QueueManager.race()`` raised (no channel left to wait on); ``error``
    is that exception. Not published when ``stop()`` ends the loop.
    """

    _role: ClassVar[Role] = Role.RUNTIME_EVENT

    error: Annotated[Any, Field(repr=False)] = None
    message: str = ""


def _settled_from(result: Any) -> TurnSettled:
    """Map a turn method's return value to a ``TurnSettled``."""
    from nooa.interactive import Done, NeedInput, Waiting

    if isinstance(result, Done):
        return TurnSettled(kind="done", result=result)
    if isinstance(result, NeedInput):
        return TurnSettled(kind="need_input", result=result)
    if isinstance(result, Waiting):
        return TurnSettled(kind="waiting", result=result)
    return TurnSettled(
        kind="error", message=f"turn returned {type(result).__name__}, not a turn result"
    )


class TurnLoop:
    """Races an agent's channels and runs one turn per wake.

    Every ``InteractiveAgent`` has one as ``agent.turns`` (hidden from the
    model, not snapshotted). Nothing runs until a host calls :meth:`start`.
    """

    def __init__(self, agent: Any) -> None:
        self._agent = agent
        self._turn_method = "handle"
        self._before_turn: BeforeTurn | None = None
        self._task: asyncio.Task[None] | None = None
        self._stopping = False
        self._turn: asyncio.Task[Any] | None = None
        self._cancel_by: str | None = None
        # The first cancelled cell output of the running turn. Handlers run
        # before the event gets its tag, so keep the event and read the tag
        # when the turn settles.
        self._interrupted: Any = None
        self._settled = asyncio.Event()
        self._settled.set()
        self._unsubscribe: Any = None

    @property
    def started(self) -> bool:
        """Whether the loop task is alive (started and not yet stopped or ended)."""
        return self._task is not None and not self._task.done()

    @property
    def running(self) -> bool:
        """Whether a turn is running now."""
        return self._turn is not None and not self._turn.done()

    def start(
        self,
        *,
        turn_method: str = "handle",
        context: contextvars.Context | None = None,
        before_turn: BeforeTurn | None = None,
    ) -> None:
        """Start the loop task; does nothing while it is already running.

        The task runs in ``context``, a fresh ``contextvars.Context`` by
        default: a loop started from inside another agent's cell must not
        inherit that agent's call stack, generation state or scoped
        blocks. Each turn task copies the loop's context.

        ``before_turn(notification)`` is awaited before each turn, after
        the items were taken from their channels. If it raises, the turn
        method is not called and the turn settles as ``error`` with
        ``ran=False``. Cancelling waits for it to finish.

        A stopped loop can be started again.
        """
        if self.started:
            return
        self._turn_method = turn_method
        self._before_turn = before_turn
        self._stopping = False
        self._settled = asyncio.Event()  # bound to this start's event loop
        self._settled.set()
        self._unsubscribe = self._agent.event_manager.on("PythonOutput", self._on_output)
        self._task = asyncio.get_running_loop().create_task(
            self._run(),
            name=f"turn-loop:{type(self._agent).__name__}",
            context=context if context is not None else contextvars.Context(),
        )

    async def cancel(self, *, by: str = "user") -> bool:
        """Stop the running turn; return whether one was running.

        Returns once the turn has settled: ``TurnCancelled`` is in the
        agent's events and every ``TurnSettled`` subscriber has run. A
        turn still in ``before_turn`` is cancelled once that returns.
        Queued items are kept and the loop goes on.
        """
        if self._settled.is_set():
            return False  # no turn in progress
        self._cancel_by = by
        turn = self._turn
        if turn is not None and not turn.done():
            turn.cancel()
        await self._settled.wait()
        return True

    def stop_starting(self) -> None:
        """Start no new turn from now on; a running turn goes on.

        For a host that is about to shut down but still has work to do
        first (closing child sessions, say): the loop no longer takes
        items, so whatever is queued stays queued. :meth:`stop` follows.
        """
        self._stopping = True

    async def stop(self, *, by: str = "host") -> None:
        """Cancel a running turn (as ``by``), then end the loop. Idempotent.

        Once called, the loop starts no new turn: items still queued stay
        in their channels.
        """
        self.stop_starting()
        await self.cancel(by=by)
        task = self._task
        if task is not None and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None

    # ---- the loop ----------------------------------------------------

    def _publish(self, event: EventBase) -> None:
        try:
            self._agent.event_manager.add(event)
        except Exception:
            logger.exception("TurnLoop: could not publish %s", event.event_type)

    def _on_output(self, event: Any) -> None:
        from nooa.events import ResultStatus

        if (
            event.execution_status is ResultStatus.CANCELLED
            and self.running
            and self._interrupted is None
        ):
            self._interrupted = event

    async def _run(self) -> None:
        queues = self._agent.queue_manager
        while not self._stopping:
            try:
                wins = await queues.race()
            except asyncio.CancelledError:
                current = asyncio.current_task()
                if current is not None and current.cancelling():
                    raise  # the loop itself is being stopped
                # A raced channel was flushed or removed, which cancels its
                # waiters: race again over the channels that are left.
                continue
            except Exception as exc:
                if self._stopping:
                    return
                # No channel left to wait on (race() raises ValueError).
                logger.exception("TurnLoop: cannot wait for input")
                self._publish(TurnLoopEnded(error=exc, message=f"{type(exc).__name__}: {exc}"))
                return
            notification: dict[str, list[Any]] = {}
            for name, item in wins:
                notification.setdefault(name, []).append(item)
            for name, channel in queues.channels().items():
                if drained := channel.drain():
                    notification.setdefault(name, []).extend(drained)
            if not notification and not any(
                channel.mode == "event" for channel in queues.channels().values()
            ):
                continue  # a wake with nothing to hand over
            await self._run_turn(notification)

    async def _run_turn(self, notification: dict[str, list[Any]]) -> None:
        self._settled.clear()
        self._cancel_by = None
        self._interrupted = None
        settled: TurnSettled | None = None
        try:
            settled = await self._turn_outcome(notification)
        finally:
            # However the turn ended (even the loop being torn down, which
            # re-raises above), cancel() must never wait forever.
            self._turn = None
            self._interrupted = None
            if settled is not None:
                self._publish(settled)
            self._settled.set()

    async def _turn_outcome(self, notification: dict[str, list[Any]]) -> TurnSettled:
        if self._before_turn is not None:
            try:
                await self._before_turn(notification)
            except Exception as exc:
                logger.exception("TurnLoop: before_turn failed; the turn does not run")
                return TurnSettled(
                    kind="error", error=exc, message=f"{type(exc).__name__}: {exc}", ran=False
                )
            if self._cancel_by is not None:
                # cancel() came while before_turn ran: the turn never starts,
                # so there is nothing to tell the model.
                return TurnSettled(kind="cancelled", cancelled_by=self._cancel_by, ran=False)
        self._publish(TurnBegan(notification=notification))
        method = getattr(self._agent, self._turn_method)
        self._turn = asyncio.create_task(method(notification), name="turn")
        try:
            return _settled_from(await self._turn)
        except asyncio.CancelledError as exc:
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                raise  # the loop itself is being torn down: not a turn outcome
            if self._cancel_by is not None:
                return self._cancelled()
            return TurnSettled(kind="error", error=exc, message="turn was cancelled from inside")
        except Exception as exc:
            logger.exception("TurnLoop: turn failed")
            return TurnSettled(kind="error", error=exc, message=f"{type(exc).__name__}: {exc}")

    def _cancelled(self) -> TurnSettled:
        """Record ``TurnCancelled`` for the model and settle as cancelled."""
        by = self._cancel_by or "host"
        interrupted = self._interrupted.tag if self._interrupted is not None else None
        try:
            self._agent.event_manager.add(TurnCancelled(by=by, interrupted=interrupted))
        except Exception as exc:
            logger.exception("TurnLoop: could not record TurnCancelled")
            return TurnSettled(
                kind="error",
                error=exc,
                message=f"the turn was cancelled but recording it failed: {type(exc).__name__}: {exc}",
            )
        return TurnSettled(kind="cancelled", cancelled_by=by, interrupted=interrupted)
