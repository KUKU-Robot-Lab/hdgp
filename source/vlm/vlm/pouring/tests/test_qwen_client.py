from __future__ import annotations

import json

import pytest

from vlm.pouring.qwen_client import QwenTaskClient, TaskGroundingUnavailable


def test_client_turns_timeout_into_explicit_unavailable_error() -> None:
    def timeout_transport(request, timeout):
        raise TimeoutError("timed out")

    client = QwenTaskClient(transport=timeout_transport)

    with pytest.raises(TaskGroundingUnavailable, match="timed out"):
        client.ground("pour", b"image")


def test_client_rejects_non_200_without_guessing_task() -> None:
    def failed_transport(request, timeout):
        return 503, b'{"detail":"model unavailable"}'

    client = QwenTaskClient(transport=failed_transport)

    with pytest.raises(TaskGroundingUnavailable, match="503"):
        client.ground("pour", b"image")


def test_client_rejects_invalid_success_body() -> None:
    def invalid_transport(request, timeout):
        return 200, json.dumps({"task": "pour", "joint_command": [1]}).encode()

    with pytest.raises(TaskGroundingUnavailable, match="invalid response"):
        QwenTaskClient(transport=invalid_transport).ground("pour", b"image")


def test_client_rejects_empty_command_or_image_before_transport() -> None:
    def unreachable_transport(request, timeout):
        raise AssertionError("transport must not be called")

    client = QwenTaskClient(transport=unreachable_transport)
    with pytest.raises(ValueError):
        client.ground("", b"image")
    with pytest.raises(ValueError):
        client.ground("pour", b"")
