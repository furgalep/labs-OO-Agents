# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""nooa_bench.gym: the thin NeMo Gym harness pieces, without Gym."""

from __future__ import annotations

import asyncio
import json
import socket
from uuid import uuid4

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from nooa_bench.gym import (
    GymEpisode,
    ModelCallRecord,
    OutputTokenLimitExceeded,
    PolicyCallBudgetExceeded,
    ShellAgent,
    ToolCallRecord,
    ToolLoopStrategy,
    _classify,
    _episode_items,
    resource_tools,
    run_gym_episode,
)

from nooa import Agent, strategy
from nooa.errors import IterationLimitExceeded, LoopDetectedError, RetryLimitExceeded
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

TRAIN = {"prompt_token_ids": [1, 2], "generation_token_ids": [3], "generation_log_probs": [-0.5]}


# --- ToolLoopStrategy -------------------------------------------------------------------------


class Counter(Agent):
    def __init__(self, llm=None, **kwargs):
        super().__init__(llm=llm, **kwargs)
        self.calls: list[str] = []

    async def shell(self, command: str) -> str:
        """Run a command."""
        self.calls.append(command)
        if command == "boom":
            raise RuntimeError("exit 7")
        return f"ran {command}"

    @strategy(ToolLoopStrategy(("shell",), max_iterations=3))
    async def solve(self, task: str) -> str:
        """Do the task."""
        ...


def _tool_turn(name: str, **arguments) -> LLMResponse:
    return LLMResponse(
        content="",
        tool_calls=[ToolCall(id=f"call_{uuid4().hex}", name=name, arguments=json.dumps(arguments))],
        finish_reason="tool_calls",
    )


async def test_tool_loop_executes_tools_then_returns_text():
    llm = FakeLLMClient(
        [
            _tool_turn("shell", command="ls"),
            _tool_turn("shell", command="boom"),
            LLMResponse(content="done"),
        ]
    )
    agent = Counter(llm=llm)
    assert await agent.solve("task") == "done"
    assert agent.calls == ["ls", "boom"]
    tool_events = list(agent.events.query(type="ToolCallEvent"))
    assert [e.result.content for e in tool_events] == ["ran ls", "exit 7"]
    assert [str(e.result.result_status) for e in tool_events] == ["complete", "error"]


async def test_tool_loop_stops_at_iteration_limit():
    llm = FakeLLMClient([_tool_turn("shell", command="ls")] * 4)
    with pytest.raises(IterationLimitExceeded):
        await Counter(llm=llm).solve("task")


async def test_unknown_tool_is_reported_to_the_model_not_raised():
    llm = FakeLLMClient([_tool_turn("rm_rf"), LLMResponse(content="ok")])
    agent = Counter(llm=llm)
    assert await agent.solve("task") == "ok"
    event = agent.events.query(type="ToolCallEvent")[0]
    assert "unknown tool" in event.result.content


# --- stop reasons -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (PolicyCallBudgetExceeded("x"), "policy_budget_exceeded"),
        (OutputTokenLimitExceeded("x"), "max_output_tokens_exceeded"),
        (IterationLimitExceeded("x"), "max_iterations_exceeded"),
        (LoopDetectedError("x"), "loop_detected"),
        (RetryLimitExceeded("x"), "retry_limit_exceeded"),
        (ValueError("x"), "error"),
    ],
)
def test_classify_direct(error, reason):
    assert _classify(error)[0] == reason


def test_classify_walks_cause_chain():
    try:
        try:
            raise IterationLimitExceeded("inner")
        except IterationLimitExceeded as inner:
            raise RuntimeError("turn failed") from inner
    except RuntimeError as outer:
        assert _classify(outer) == ("max_iterations_exceeded", "inner")


# --- episode projection -----------------------------------------------------------------------


def _record(request_input, output, status=200):
    return ModelCallRecord(
        invocation_id="inv",
        request={"input": request_input},
        response={"id": "resp", "output": output} if status == 200 else {"error": {}},
        status_code=status,
    )


def test_episode_items_keep_typed_items_and_full_conversation():
    history = [
        {"role": "user", "content": "task"},
        {"type": "function_call", "call_id": "c1", "name": "shell", "arguments": "{}", **TRAIN},
        {"type": "function_call_output", "call_id": "c1", "output": "ran"},
        {"role": "user", "content": "<context/>"},
    ]
    final = [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "done"}],
            **TRAIN,
        }
    ]
    output, conversation = _episode_items(
        [_record([{"role": "user", "content": "task"}], [], 500), _record(history, final)]
    )
    assert [i.get("type") for i in output] == ["function_call", "function_call_output", "message"]
    assert conversation == [*history, *final]
    assert output[0]["generation_token_ids"] == [3]


def test_episode_items_without_a_finished_response():
    assert _episode_items([_record([{"role": "user", "content": "t"}], [], 400)]) == ([], [])


def test_episode_json_round_trip():
    episode = GymEpisode(
        output_items=[{"type": "message"}],
        conversation=[],
        model_calls=[_record([], [])],
        tool_calls=[
            ToolCallRecord("inv", "c1", "shell", {"command": "ls"}, "ran", 1.0, 2.0, "completed")
        ],
        return_value={"value": 1},
        stop_reason="completed",
        resource_cookies={"sid": "1"},
    )
    restored = GymEpisode.from_json(episode.to_json())
    assert restored == episode


# --- resource tools ---------------------------------------------------------------------------


def test_resource_tools_build_schema_and_validate(monkeypatch):
    seen = {}

    async def fake_post(self, url, json=None, headers=None):
        seen["url"], seen["json"], seen["headers"] = url, json, headers
        request = httpx.Request("POST", url)
        return httpx.Response(
            200, json={"ok": True}, request=request, headers={"set-cookie": "rs=abc"}
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    cookies: dict[str, str] = {}
    [tool] = resource_tools(
        [
            {
                "type": "function",
                "name": "increment",
                "description": "Add to the counter.",
                "parameters": {
                    "type": "object",
                    "properties": {"by": {"type": "integer"}},
                    "required": ["by"],
                },
            }
        ],
        base_url="http://resources/ng-rollout/r/",
        headers={"x-grant": "1"},
        cookies=cookies,
    )
    assert tool.get_parameter_schema()["properties"]["by"]["type"] == "integer"
    assert asyncio.run(tool.callable(by="three")) == {
        "error": "Invalid arguments for increment: 'three' is not of type 'integer'"
    }
    assert asyncio.run(tool.callable(by=3)) == {"ok": True}
    assert seen["url"] == "http://resources/ng-rollout/r/increment" and seen["json"] == {"by": 3}
    assert seen["headers"] == {"x-grant": "1"}


def test_resource_tools_reject_non_function_tools():
    with pytest.raises(ValueError):
        resource_tools([{"type": "web_search"}], base_url="http://r")


# --- run_gym_episode against a Gym-shaped Responses server ------------------------------------


def _responses_app(log: list[dict]) -> FastAPI:
    app = FastAPI()
    state = {"n": 0}

    @app.post("/ng-rollout/{rollout_id}/v1/responses")
    async def responses(rollout_id: str, request: Request):
        body = await request.json()
        log.append({"headers": dict(request.headers), "body": body})
        state["n"] += 1
        if state["n"] == 1:
            output = [
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "shell",
                    "status": "completed",
                    "arguments": json.dumps({"command": "echo hi"}),
                    **TRAIN,
                }
            ]
        else:
            output = [
                {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "DONE", "annotations": []}],
                    **TRAIN,
                }
            ]
        payload = {
            "id": f"resp_{state['n']}",
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "error": None,
            "incomplete_details": None,
            "model": "gym-policy",
            "output": output,
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        }
        response = JSONResponse(payload)
        response.set_cookie("gym_model_session", "s1")
        return response

    return app


async def _serve(app: FastAPI):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    while not server.started:
        await asyncio.sleep(0.01)
    return server, task, sock.getsockname()[1]


async def test_run_gym_episode_records_wire_and_tools():
    log: list[dict] = []
    server, task, port = await _serve(_responses_app(log))
    commands: list[str] = []

    async def run_command(command: str) -> str:
        commands.append(command)
        return "hi\n[exit 0]"

    try:
        episode = await run_gym_episode(
            lambda llm: ShellAgent(llm=llm, run_command=run_command),
            lambda agent: agent.solve("say hi"),
            model_url=f"http://127.0.0.1:{port}/ng-rollout/r1/v1",
            max_policy_calls=5,
        )
    finally:
        server.should_exit = True
        await task
    assert episode.stop_reason == "completed" and episode.return_value == "DONE"
    assert commands == ["echo hi"]
    assert [r.status_code for r in episode.model_calls] == [200, 200]
    assert [i["type"] for i in episode.output_items] == [
        "function_call",
        "function_call_output",
        "message",
    ]
    assert episode.output_items[0]["generation_token_ids"] == [
        3
    ]  # replayed verbatim with training fields
    assert [t.name for t in episode.tool_calls] == ["shell"]
    assert "x-session-id" in log[0]["headers"]
    assert "gym_model_session=s1" in log[1]["headers"].get("cookie", "")


async def test_run_gym_episode_policy_budget_is_a_gradable_stop():
    log: list[dict] = []
    server, task, port = await _serve(_responses_app(log))
    try:
        episode = await run_gym_episode(
            lambda llm: ShellAgent(llm=llm, run_command=lambda c: asyncio.sleep(0, result="ok")),
            lambda agent: agent.solve("say hi"),
            model_url=f"http://127.0.0.1:{port}/ng-rollout/r1/v1",
            max_policy_calls=1,
        )
    finally:
        server.should_exit = True
        await task
    assert episode.stop_reason == "policy_budget_exceeded"
    assert len(log) == 1
