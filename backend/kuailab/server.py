from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from .config import Settings
from .engine import CampaignEngine
from .state import StateStore


settings = Settings()
store = StateStore(settings.state_dir, settings.public_dict())
def refresh_runtime_config(state: dict) -> None:
    state["config"] = settings.public_dict()
    if state.get("iterations") and state.get("campaign", {}).get("mode") == "demo":
        state["iterations"][0]["title"] = "Demo FM baseline"
        state["iterations"][0]["provider"] = "demo"
        state["iterations"][0]["evidence"] = "synthetic-demo"


store.update(refresh_runtime_config)
engine = CampaignEngine(settings, store)


class Handler(BaseHTTPRequestHandler):
    server_version = "KuaiLab/2.0"

    def _origin(self) -> str:
        origin = self.headers.get("Origin", "")
        return origin if origin in {"http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3001", "http://127.0.0.1:3001"} else "http://localhost:3000"

    def _headers(self, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", self._origin())
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def _json(self, data: dict, status: int = 200) -> None:
        self._headers(status)
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def _body(self) -> dict:
        length = min(int(self.headers.get("Content-Length", "0")), 20_000)
        if not length:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_OPTIONS(self) -> None:
        self._headers(HTTPStatus.NO_CONTENT)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/health":
            self._json({"ok": True, "version": "2.0.0", "engine_running": engine.running, "config": settings.public_dict()})
        elif path == "/api/state":
            self._json(store.snapshot())
        elif path == "/api/events":
            snapshot = store.snapshot()
            self._json({"events": snapshot["events"]})
        else:
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            body = self._body()
            if path == "/api/run/start":
                engine.start(body.get("mode", "demo"), body.get("provider", "demo"))
            elif path == "/api/run/pause":
                engine.pause()
            elif path == "/api/run/resume":
                engine.resume()
            elif path == "/api/run/stop":
                engine.stop()
            elif path == "/api/run/reset":
                engine.reset()
            elif path == "/api/steer":
                engine.steer(str(body.get("instruction", "")))
            else:
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                return
            self._json({"ok": True, "state": store.snapshot()})
        except (ValueError, RuntimeError, json.JSONDecodeError) as error:
            self._json({"ok": False, "error": str(error)}, HTTPStatus.CONFLICT)
        except Exception as error:
            self._json({"ok": False, "error": f"internal error: {str(error)[:300]}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, format: str, *args) -> None:
        return


def main() -> None:
    server = ThreadingHTTPServer((settings.host, settings.port), Handler)
    print(f"KuaiLab backend http://{settings.host}:{settings.port}", flush=True)
    print(f"Model {settings.model} · key {'available' if settings.api_key_available else 'missing'}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
