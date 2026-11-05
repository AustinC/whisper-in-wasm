#!/usr/bin/env python3
"""Serves web/ on http://localhost:8080 (localhost counts as a secure
context, so the microphone and Cache Storage work)."""
import functools, http.server, os, sys

port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "web")

class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                      ".wasm": "application/wasm", ".js": "text/javascript"}
    def end_headers(self):
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

http.server.ThreadingHTTPServer(("127.0.0.1", port),
    functools.partial(Handler, directory=root)).serve_forever()
