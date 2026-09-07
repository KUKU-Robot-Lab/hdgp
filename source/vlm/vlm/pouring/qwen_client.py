from __future__ import annotations

import base64
import json
from collections.abc import Callable
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .contracts import TaskSpecification
from .task_grounding import parse_task_specification

Transport = Callable[[Request, float], tuple[int, bytes]]


class TaskGroundingUnavailable(RuntimeError):
    """The task-grounding service did not return a trustworthy task."""


def _default_transport(request: Request, timeout: float) -> tuple[int, bytes]:
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except HTTPError as exc:
        return exc.code, exc.read()


class QwenTaskClient:
    """Bounded localhost client kept outside the control loop."""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8100/v1/task-grounding",
        timeout_seconds: float = 30.0,
        transport: Transport | None = None,
    ) -> None:
        if timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be positive")
        self.base_url = base_url
        self.timeout_seconds = float(timeout_seconds)
        self.transport = transport or _default_transport

    def ground(self, command: str, image: bytes) -> TaskSpecification:
        if not command.strip():
            raise ValueError("command must be non-empty")
        if not image:
            raise ValueError("image must be non-empty")
        payload = json.dumps(
            {
                "command": command,
                "image_base64": base64.b64encode(image).decode("ascii"),
            }
        ).encode("utf-8")
        request = Request(
            self.base_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            status, body = self.transport(request, self.timeout_seconds)
        except (TimeoutError, OSError) as exc:
            raise TaskGroundingUnavailable(str(exc)) from exc
        if status != 200:
            raise TaskGroundingUnavailable(f"task-grounding service returned HTTP {status}")
        try:
            decoded = json.loads(body.decode("utf-8"))
            if not isinstance(decoded, dict):
                raise ValueError("response root is not an object")
            return parse_task_specification(json.dumps(decoded))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise TaskGroundingUnavailable(f"invalid response: {exc}") from exc
