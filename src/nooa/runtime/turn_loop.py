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
"""

import asyncio
import contextvars
import logging
from contextlib import suppress
from typing import Annotated, Any, ClassVar, Literal

from pydantic import Field

from nooa.context_blocks import EventBase
from nooa.context_blocks.roles import Role

logger = logging.getLogger(__name__)

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
      ``None``). ``TurnCancelled`` is already in the agent's events.
    - ``error``: ``message`` says what went wrong and ``error`` is the
      exception (``None`` for a turn that returned something other than a
      turn result).

    A runtime event: never recorded, never shown to the model.
    """

    _role: ClassVar[Role] = Role.RUNTIME_EVENT

    kind: TurnKind
    result: Annotated[Any, Field(repr=False)] = None
    error: Annotated[Any, Field(repr=False)] = None
    message: str = ""
    cancelled_by: str | None = None
    interrupted: str | None = None


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
        """Whether :meth:`start` has run and :meth:`stop` has not."""
        return self._task is not None

    @property
    def running(self) -> bool:
        """Whether a turn is running now."""
        return self._turn is not None and not self._turn.done()

    def start(
        self, *, turn_method: str = "handle", context: contextvars.Context | None = None
    ) -> None:
        """Start the loop task; a second call does nothing.

        The task runs in ``context``, a fresh ``contextvars.Context`` by
        default: a loop started from inside another agent's cell must not
        inherit that agent's call stack, generation state or scoped
        blocks. Each turn task copies the loop's context.
        """
        if self._task is not None:
            return
        self._turn_method = turn_method
        self._unsubscribe = self._agent.event_manager.on("PythonOutput", self._on_output)
        self._task = asyncio.get_running_loop().create_task(
            self._run(),
            name=f"turn-loop:{type(self._agent).__name__}",
            context=context if context is not None else contextvars.Context(),
        )

    async def cancel(self, *, by: str = "user") -> bool:
        """Stop the running turn; return whether one was running.

        Returns once the turn has settled: ``TurnCancelled`` is in the
        agent's events and every ``TurnSettled`` subscriber has run.
        Queued items are kept and the loop goes on.
        """
        turn = self._turn
        if turn is None or turn.done():
            return False
        self._cancel_by = by
        turn.cancel()
        await self._settled.wait()
        return True

    async def stop(self, *, by: str = "host") -> None:
        """Cancel a running turn (as ``by``), then end the loop. Idempotent.

        Once called, the loop starts no new turn: items still queued stay
        in their channels.
        """
        self._stopping = True
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
        self._agent.event_manager.add(event)

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
        self._publish(TurnBegan(notification=notification))
        method = getattr(self._agent, self._turn_method)
        self._turn = asyncio.create_task(method(notification), name="turn")
        try:
            settled = _settled_from(await self._turn)
        except asyncio.CancelledError as exc:
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                # The loop itself is being stopped: not a turn outcome.
                self._turn = None
                self._settled.set()
                raise
            if self._cancel_by is not None:
                interrupted = self._interrupted.tag if self._interrupted is not None else None
                self._agent.event_manager.add(
                    TurnCancelled(by=self._cancel_by, interrupted=interrupted)
                )
                settled = TurnSettled(
                    kind="cancelled", cancelled_by=self._cancel_by, interrupted=interrupted
                )
            else:
                settled = TurnSettled(
                    kind="error", error=exc, message="turn was cancelled from inside"
                )
        except Exception as exc:
            logger.exception("TurnLoop: turn failed")
            settled = TurnSettled(kind="error", error=exc, message=f"{type(exc).__name__}: {exc}")
        finally:
            self._turn = None
            self._interrupted = None
        self._publish(settled)
        self._settled.set()
