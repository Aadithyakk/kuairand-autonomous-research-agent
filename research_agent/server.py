from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from .core import ResearchController


class AgentAPI:
    def __init__(self, controller: ResearchController):
        self.controller = controller

    def handler(self) -> type[BaseHTTPRequestHandler]:
        api = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "KuaiRandResearchAgent/0.1"

            def log_message(self, format: str, *args: Any) -> None:
                return

            def _headers(self, status: int = HTTPStatus.OK) -> None:
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.end_headers()

            def _json(self, value: Any, status: int = HTTPStatus.OK) -> None:
                self._headers(status)
                self.wfile.write(json.dumps(value, ensure_ascii=False).encode("utf-8"))

            def _body(self) -> dict[str, Any]:
                length = int(self.headers.get("Content-Length", "0"))
                if length == 0:
                    return {}
                return json.loads(self.rfile.read(length).decode("utf-8"))

            def do_OPTIONS(self) -> None:  # noqa: N802
                self._headers(HTTPStatus.NO_CONTENT)

            def do_GET(self) -> None:  # noqa: N802
                path = urlparse(self.path).path
                if path == "/api/health":
                    self._json({"ok": True, "service": "kuairand-research-agent"})
                elif path == "/api/state":
                    self._json(api.controller.initialize())
                elif path == "/api/literature":
                    self._json(api.controller.literature.cards)
                elif path == "/api/actions":
                    self._json(api.controller.actions)
                else:
                    self._json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

            def do_POST(self) -> None:  # noqa: N802
                path = urlparse(self.path).path
                try:
                    body = self._body()
                    if path == "/api/run/start":
                        self._json(api.controller.start())
                    elif path == "/api/run/pause":
                        self._json(api.controller.pause())
                    elif path == "/api/run/resume":
                        self._json(api.controller.resume())
                    elif path == "/api/run/stop":
                        self._json(api.controller.stop())
                    elif path == "/api/steer":
                        self._json(api.controller.steer(str(body.get("message", ""))))
                    elif path == "/api/reset":
                        self._json(api.controller.initialize(force=True))
                    else:
                        self._json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
                except (ValueError, json.JSONDecodeError) as exc:
                    self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                except Exception as exc:
                    self._json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

        return Handler


def serve(controller: ResearchController, host: str = "127.0.0.1", port: int = 8765) -> None:
    controller.initialize()
    server = ThreadingHTTPServer((host, port), AgentAPI(controller).handler())
    print(f"Research agent API listening on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
