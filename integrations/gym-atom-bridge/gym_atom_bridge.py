# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the pinned Atom session host with Gym's model and lifecycle hooks."""

from __future__ import annotations

from pathlib import Path
from tempfile import mkdtemp

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nooa_atom.session.items import TaskResult

from nooa import Agent
from nooa.interactive import Done
from nooa.runtime.hooks import compose_hooks, get_hooks, set_hooks
from nooa.tools.shell_lifecycle import BackgroundServiceScope, preserve_background_services
from nooa_atom.agent.factory import create_session_agent
from nooa_atom.session.loader import ATOM_AGENT
from nooa_atom.session.options import SessionOptions
from nooa_atom.session.registry import SessionRegistry
from nooa_atom.session.store import SessionStore, sessions_root

from responses_api_agents.nooa_agent.task_agent import _latest_user_text


class GymAtomAgent(Agent):
    """Gym construction shell; the actual task agent is Atom's native session."""


def adopt_session_shells(shells: BackgroundServiceScope, session_agent: object) -> int:
    """Adopt every distinct bash session held by an Atom agent; return how many."""
    adopted: dict[int, object] = {}
    for holder in (
        getattr(session_agent, "_base_shell", None),
        getattr(session_agent, "shell", None),
    ):
        bash = getattr(holder, "session", None)
        if bash is not None and hasattr(bash, "_keep_background_on_close"):
            adopted[id(bash)] = bash
    for bash in adopted.values():
        shells.adopt(bash)
    return len(adopted)


async def invoke_atom(
    agent: GymAtomAgent, request: NeMoGymResponseCreateParamsNonStreaming
) -> TaskResult:
    from nooa import tracing

    # Services the agent starts (``nohup ... &``, daemons) are graded after this
    # invocation returns. Gym's supervisor owns and reaps them after verification,
    # so every Atom shell must close without killing its background jobs.
    async with preserve_background_services() as shells:
        # These processes belong to one rollout. Keep the complete native trace,
        # including code execution, for export before the sandbox is destroyed.
        trace_dir = Path(mkdtemp(prefix="gym-atom-native-"))
        tracing.enable_tracing(exporters=[tracing.exporters.jsonl(str(trace_dir))])
        gym_hooks = get_hooks()

        class ObservedRegistry(SessionRegistry):
            async def _build(self, options, handle, *, restore=False):
                session = await super()._build(options, handle, restore=restore)

                def enter():
                    # Atom deliberately starts each session in a fresh Context.
                    # Reinstall Gym capture for roots AND delegated children.
                    set_hooks(compose_hooks(get_hooks(), gym_hooks))
                    tracing.set_session(session.id)
                    tracing.register_hooks_in_current_context()

                session.add_loop_context_hook(enter)
                return session

        def adopting_factory(options, storage):
            # SessionRegistry runs the factory for the root and every delegated child
            # in a fresh contextvars.Context, so BashSession's automatic adoption by the
            # invocation scope cannot see these shells. Adopt each one explicitly.
            session_agent = create_session_agent(options, storage)
            adopt_session_shells(shells, session_agent)
            return session_agent

        def same_model(alias, workspace):
            raise ValueError("This Gym rollout permits only its registered policy model")

        options = SessionOptions(
            workspace=Path.cwd(),
            agent_spec=ATOM_AGENT,
            llm=agent.llm,
            turn_method="handle_batch",
            sessions_dir=trace_dir / "sessions",
        )
        store = SessionStore(sessions_root(options.workspace, options.sessions_dir))
        registry = ObservedRegistry(store, agent_factory=adopting_factory, llm_factory=same_model)
        try:
            root = await registry.create(options)
            outcome = await root.prompt(_latest_user_text(request))
            if not isinstance(outcome, Done) or outcome.result is None:
                raise RuntimeError(f"Atom ended without a task result: {type(outcome).__name__}")
            return outcome.result
        finally:
            # Inside the scope: adopted shells close through the preserving path.
            await registry.close_all()
            tracing.shutdown_traces()
