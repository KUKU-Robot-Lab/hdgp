from __future__ import annotations

import base64
import json
from threading import Thread

import pytest

from vlm.pouring.qwen_client import QwenTaskClient
from vlm.pouring.qwen_server import (
    ModelResponseError,
    RequestValidationError,
    TaskGroundingService,
    create_server,
)


class FakeBackend:
    model_id = "fake/qwen"
    loaded = True

    def generate(self, command: str, image: bytes) -> str:
        assert command
        assert image
        return json.dumps(
            {
                "task": "pour",
                "source_id": "right_cup",
                "target_id": "left_cup",
                "nominal_plan": ["grasp_lift", "pre_pour_bridge", "bimanual_pour"],
                "allowed_skills": ["grasp_lift", "pre_pour_bridge", "bimanual_pour", "recovery"],
            }
        )


def test_service_returns_validated_specification_and_health() -> None:
    service = TaskGroundingService(FakeBackend())

    result = service.ground(
        "Pour from the right cup into the left cup",
        base64.b64encode(b"image").decode("ascii"),
    )

    assert result["source_id"] == "right_cup"
    assert service.health() == {"model_id": "fake/qwen", "loaded": True}


@pytest.mark.parametrize(
    ("command", "image_base64"),
    [
        ("", base64.b64encode(b"image").decode("ascii")),
        ("pour", "%%%not-base64%%%"),
        ("pour", ""),
    ],
)
def test_service_rejects_invalid_request(command: str, image_base64: str) -> None:
    with pytest.raises(RequestValidationError):
        TaskGroundingService(FakeBackend()).ground(command, image_base64)


def test_service_rejects_malformed_or_control_model_output() -> None:
    class BadBackend(FakeBackend):
        def __init__(self, output: str) -> None:
            self.output = output

        def generate(self, command: str, image: bytes) -> str:
            return self.output

    for output in ("not-json", '{"task":"pour","joint_command":[1]}'):
        with pytest.raises(ModelResponseError):
            TaskGroundingService(BadBackend(output)).ground(
                "pour",
                base64.b64encode(b"image").decode("ascii"),
            )


def test_localhost_server_and_client_round_trip() -> None:
    server = create_server(FakeBackend(), host="127.0.0.1", port=0)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = QwenTaskClient(
            base_url=f"http://127.0.0.1:{server.server_port}/v1/task-grounding",
            timeout_seconds=2.0,
        )
        result = client.ground("pour", b"image")
        assert result.target_id == "left_cup"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)
