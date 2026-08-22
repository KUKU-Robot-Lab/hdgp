from __future__ import annotations

import base64
import json
from urllib.request import Request

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


def test_service_and_client_round_trip_the_http_json_contract() -> None:
    service = TaskGroundingService(FakeBackend())

    def service_transport(request: Request, timeout: float) -> tuple[int, bytes]:
        assert timeout == 2.0
        assert isinstance(request.data, bytes)
        payload = json.loads(request.data)
        result = service.ground(payload["command"], payload["image_base64"])
        return 200, json.dumps(result).encode("utf-8")

    client = QwenTaskClient(timeout_seconds=2.0, transport=service_transport)

    result = client.ground("pour", b"image")

    assert result.target_id == "left_cup"


def test_create_server_uses_requested_local_address(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_server(address, handler):
        captured["address"] = address
        captured["handler"] = handler
        return "server"

    monkeypatch.setattr("vlm.pouring.qwen_server.ThreadingHTTPServer", fake_server)

    result = create_server(FakeBackend(), host="127.0.0.1", port=8100)

    assert result == "server"
    assert captured["address"] == ("127.0.0.1", 8100)
