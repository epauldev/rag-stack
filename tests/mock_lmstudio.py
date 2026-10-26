"""Faux serveur LM Studio (API OpenAI) pour tester la chaîne sans modèle.

- /v1/models, /v1/embeddings : vecteurs « sac de mots hachés » déterministes (les textes qui
  partagent des mots obtiennent des vecteurs proches).
- /v1/chat/completions : réponse factice qui cite [1], avec ou sans streaming.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DIM = 256
CALLS = {"embeddings": 0, "chat": 0}


def embed_text(text: str) -> list[float]:
    vec = [0.0] * DIM
    for tok in re.findall(r"\w+", text.lower()):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        vec[h % DIM] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence
        pass

    def _json(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            self._json({"object": "list", "data": [
                {"id": "mock-embed-bge-m3", "object": "model"},
                {"id": "mock-chat-model", "object": "model"},
            ]})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if self.path.endswith("/embeddings"):
            CALLS["embeddings"] += 1
            inputs = body["input"] if isinstance(body["input"], list) else [body["input"]]
            self._json({"object": "list", "model": body.get("model"), "usage": {"prompt_tokens": 1, "total_tokens": 1},
                        "data": [{"object": "embedding", "index": i, "embedding": embed_text(t)}
                                 for i, t in enumerate(inputs)]})
        elif self.path.endswith("/chat/completions"):
            CALLS["chat"] += 1
            user = body["messages"][-1]["content"]
            first = re.search(r"\[1\] (.+)", user)
            answer = "<think>raisonnement interne</think>Réponse simulée d'après " + (
                first.group(1)[:60] if first else "aucun extrait") + " [1]."
            if body.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for i in range(0, len(answer), 7):
                    chunk = {"id": "x", "object": "chat.completion.chunk", "created": 0, "model": body.get("model"),
                             "choices": [{"index": 0, "delta": {"content": answer[i:i + 7]}, "finish_reason": None}]}
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    self.wfile.flush()
                done = {"id": "x", "object": "chat.completion.chunk", "created": 0, "model": body.get("model"),
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
                self.wfile.write(f"data: {json.dumps(done)}\n\ndata: [DONE]\n\n".encode())
            else:
                self._json({"id": "x", "object": "chat.completion", "created": 0, "model": body.get("model"),
                            "choices": [{"index": 0, "finish_reason": "stop",
                                         "message": {"role": "assistant", "content": answer}}],
                            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})
        else:
            self._json({"error": "not found"}, 404)


def start(port: int = 0) -> tuple[ThreadingHTTPServer, str]:
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/v1"


if __name__ == "__main__":
    srv, url = start(1234)
    print("Mock LM Studio sur", url)
    threading.Event().wait()
