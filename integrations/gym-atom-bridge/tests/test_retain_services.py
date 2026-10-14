# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Services started by the root and a delegated child outlive invoke_atom's close_all()."""

import json
import os
import signal
import socket
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import gym_atom_bridge
import pytest
from nemo_gym.openai_utils import (
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
)
from responses_api_agents.nooa_agent.invocation import NOOAInvocationConfig
from responses_api_agents.nooa_agent.runner import InProcessNOOARunner, NOOARunRequest
from responses_api_agents.nooa_agent.tests.test_gym_llm import FakeHTTPResponse, model_response


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:
        return Path(f"/proc/{pid}/stat").read_text().split(") ")[-1][0] != "Z"
    except FileNotFoundError:
        return False


def wait_until(predicate, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def serves(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=1) as response:
            return response.status == 200
    except OSError:
        return False


async def run_services_fixture(tmp_path: Path, monkeypatch) -> tuple[int, int, int]:
    """Root starts an HTTP server, its delegated child a sleeper; both via nohup ... &."""
    monkeypatch.chdir(tmp_path)
    port = free_port()
    root_code = (
        f"r = await self.shell.run('nohup python3 -m http.server {port} --bind 127.0.0.1 "
        f">http.log 2>&1 & echo $! > root.pid')\n"
        "child = await self.delegate('start sidecar', 'start the sidecar')\n"
        "return_result(Done(explanation='root', result=TaskResult(solution_description='root', "
        "evidence=child.result.evidence, how_to_verify='offline', report='services started')))"
    )
    child_code = (
        "r = await self.shell.run('nohup sleep 300 >/dev/null 2>&1 & echo $! > child.pid')\n"
        "return_result(Done(explanation='child', result=TaskResult(solution_description='child', "
        "evidence='sidecar started', how_to_verify='offline', report='child')))"
    )
    code = [root_code, child_code]
    calls = []

    async def post(**kwargs):
        calls.append(kwargs)
        assert len(calls) <= 2
        out = NeMoGymResponseFunctionToolCall(
            id=f"f{len(calls)}",
            call_id=f"c{len(calls)}",
            name="python_cell",
            arguments=json.dumps({"code": code[len(calls) - 1]}),
        )
        return FakeHTTPResponse(model_response(out, response_id=f"r{len(calls)}"))

    runner = InProcessNOOARunner(
        invocation=NOOAInvocationConfig(
            agent_class="gym_atom_bridge:GymAtomAgent",
            invocation_adapter="gym_atom_bridge:invoke_atom",
        ),
        server_client=SimpleNamespace(post=post),
        model_server_name="policy",
        max_policy_calls=2,
        context_window=1048576,
    )
    result = await runner.run(
        NOOARunRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="start services"),
            model_url_path="/fixture/v1/responses",
        )
    )
    assert result.termination_reason is None, result.termination_error
    assert result.return_value.evidence == "sidecar started"
    assert len(calls) == 2
    # invoke_atom has returned: registry.close_all() and the scope exit have both run.
    return port, int((tmp_path / "root.pid").read_text()), int((tmp_path / "child.pid").read_text())


def kill(*pids: int) -> None:
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


@pytest.mark.asyncio
async def test_root_and_child_services_survive_close_all(tmp_path, monkeypatch):
    port, root_pid, child_pid = await run_services_fixture(tmp_path, monkeypatch)
    try:
        assert alive(root_pid), "root's nohup http.server was killed by close_all"
        assert alive(child_pid), "delegated child's nohup service was killed by close_all"
        assert wait_until(lambda: serves(port)), "root's http.server stopped serving"
        print("retained", {"http.server": root_pid, "child sleeper": child_pid, "port": port})
    finally:
        kill(root_pid, child_pid)
    assert wait_until(lambda: not alive(root_pid) and not alive(child_pid))


@pytest.mark.asyncio
async def test_without_adoption_close_all_kills_services(tmp_path, monkeypatch):
    """Control: Atom builds agents in a fresh Context, so the scope alone does not see them."""
    monkeypatch.setattr(gym_atom_bridge, "adopt_session_shells", lambda shells, agent: 0)
    port, root_pid, child_pid = await run_services_fixture(tmp_path, monkeypatch)
    try:
        assert wait_until(lambda: not alive(root_pid) and not alive(child_pid), timeout=10)
        assert not serves(port)
    finally:
        kill(root_pid, child_pid)
