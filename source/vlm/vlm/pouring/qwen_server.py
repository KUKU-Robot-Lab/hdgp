from __future__ import annotations

import argparse
from dataclasses import asdict
import base64
import binascii
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from typing import Any, Protocol

from .qwen_backend import QwenBackend
from .task_grounding import parse_task_specification

_MAX_REQUEST_BYTES = 20 * 1024 * 1024


class GenerationBackend(Protocol):
    model_id: str
    loaded: bool

    def generate(self, command: str, image: bytes) -> str: ...


class RequestValidationError(ValueError):
    """The HTTP request does not contain a usable command and RGB image."""


class ModelResponseError(RuntimeError):
    """The model failed or emitted an invalid task contract."""


class TaskGroundingService:
    def __init__(self, backend: GenerationBackend) -> None:
        self.backend = backend

    def health(self) -> dict[str, object]:
        return {"model_id": self.backend.model_id, "loaded": self.backend.loaded}

    def ground(self, command: str, image_base64: str) -> dict[str, object]:
        if not isinstance(command, str) or not command.strip():
            raise RequestValidationError("command must be non-empty")
        if not isinstance(image_base64, str) or not image_base64:
            raise RequestValidationError("image_base64 must be non-empty")
        try:
            image = base64.b64decode(image_base64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise RequestValidationError("image_base64 must contain valid base64") from exc
        if not image:
            raise RequestValidationError("decoded image must be non-empty")
        try:
            raw = self.backend.generate(command, image)
            specification = parse_task_specification(raw)
        except Exception as exc:
            if isinstance(exc, RequestValidationError):
                raise
            raise ModelResponseError(f"model did not produce a valid TaskSpecification: {exc}") from exc
        return asdict(specification)


def _handler_type(service: TaskGroundingService) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path != "/health":
                self._write_json(404, {"detail": "not found"})
                return
            self._write_json(200, service.health())

        def do_POST(self) -> None:
            if self.path != "/v1/task-grounding":
                self._write_json(404, {"detail": "not found"})
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                if content_length <= 0 or content_length > _MAX_REQUEST_BYTES:
                    raise RequestValidationError("invalid Content-Length")
                payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
                if not isinstance(payload, dict):
                    raise RequestValidationError("request root must be an object")
                result = service.ground(payload.get("command"), payload.get("image_base64"))
            except (RequestValidationError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                self._write_json(422, {"detail": str(exc)})
                return
            except ModelResponseError as exc:
                self._write_json(502, {"detail": str(exc)})
                return
            self._write_json(200, result)

        def _write_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    return Handler


def create_server(
    backend: GenerationBackend,
    *,
    host: str = "127.0.0.1",
    port: int = 8100,
) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), _handler_type(TaskGroundingService(backend)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the same-machine Qwen task-grounding service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    args = parser.parse_args()
    server = create_server(QwenBackend(args.model), host=args.host, port=args.port)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
