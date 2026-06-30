#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Innate Inc

"""
Brain Debug Panel

A sleek, real-time debug panel for monitoring the Brain's internal state.
This module provides a web-based interface to visualize:
- Brain state (connection, model, directive)
- Primitive execution status
- History entries
- Message queue
- Discrepancies

Usage:
    # Import and use in your server
    from src.debug_panel import DebugPanelServer, register_brain_for_debug

    # Register a brain instance for debugging
    register_brain_for_debug(brain_instance)

    # Start the debug panel server
    debug_server = DebugPanelServer(port=8081)
    await debug_server.start()
"""

import json
import threading
import time
import weakref
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Dict, Optional

# Directory containing static assets
STATIC_DIR = Path(__file__).parent / "debug_panel_static"

# Global registry for brain instances
_brain_registry: Dict[str, weakref.ref] = {}
_registry_lock = threading.Lock()


def register_brain_for_debug(brain, connection_id: Optional[str] = None):
    """Register a brain instance for debugging."""
    with _registry_lock:
        cid = connection_id or brain.connection_id
        _brain_registry[cid] = weakref.ref(brain)


def unregister_brain_for_debug(connection_id: str):
    """Unregister a brain instance from debugging."""
    with _registry_lock:
        _brain_registry.pop(connection_id, None)


def get_all_brain_states() -> Dict[str, dict]:
    """Get debug state from all registered brains."""
    states = {}
    with _registry_lock:
        dead_keys = []
        for cid, brain_ref in _brain_registry.items():
            brain = brain_ref()
            if brain is None:
                dead_keys.append(cid)
            else:
                try:
                    states[cid] = brain.get_debug_state()
                except Exception as e:
                    states[cid] = {"error": str(e)}

        # Clean up dead references
        for key in dead_keys:
            del _brain_registry[key]

    return states


# Cache for assembled HTML (loaded once at startup)
_html_cache: Optional[str] = None


def get_debug_panel_html() -> str:
    """
    Load and return the debug panel HTML with inlined CSS/JS.

    Static assets are loaded from the debug_panel_static directory and
    inlined into a single HTML response for easy serving.
    """
    global _html_cache

    if _html_cache is not None:
        return _html_cache

    # Load static files
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    css = (STATIC_DIR / "styles.css").read_text(encoding="utf-8")
    js = (STATIC_DIR / "script.js").read_text(encoding="utf-8")

    # Inline CSS and JS into HTML
    html = html.replace("<!-- INJECT_CSS -->", f"<style>\n{css}\n</style>")
    html = html.replace("<!-- INJECT_JS -->", f"<script>\n{js}\n</script>")

    _html_cache = html
    return html


def reload_static_assets():
    """Force reload of static assets (useful during development)."""
    global _html_cache
    _html_cache = None


class DebugPanelHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the debug panel."""

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self._serve_html()
        elif self.path == "/api/debug":
            self._serve_debug_api()
        elif self.path == "/reload":
            self._handle_reload()
        else:
            self._serve_404()

    def _serve_html(self):
        """Serve the debug panel HTML page."""
        self.send_response(200)
        self.send_header("Content-type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(get_debug_panel_html().encode("utf-8"))

    def _serve_debug_api(self):
        """Serve the debug state as JSON."""
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        states = get_all_brain_states()
        self.wfile.write(json.dumps(states).encode("utf-8"))

    def _handle_reload(self):
        """Reload static assets (development helper)."""
        reload_static_assets()
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(b"Static assets reloaded")

    def _serve_404(self):
        """Serve a 404 response."""
        self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress HTTP logging


class DebugPanelServer:
    """
    HTTP server for the Brain debug panel.

    Usage:
        server = DebugPanelServer(port=8081)
        server.start()
        # ... later ...
        server.stop()
    """

    def __init__(self, port: int = 8081, host: str = "0.0.0.0"):
        self.port = port
        self.host = host
        self.server: Optional[HTTPServer] = None
        self.server_thread: Optional[threading.Thread] = None

    def start(self):
        """Start the debug panel server in a background thread."""
        self.server = HTTPServer((self.host, self.port), DebugPanelHandler)
        self.server_thread = threading.Thread(
            target=self.server.serve_forever, daemon=True
        )
        self.server_thread.start()
        print(f"[DebugPanel] Started at http://{self.host}:{self.port}")

    def stop(self):
        """Stop the debug panel server."""
        if self.server:
            self.server.shutdown()
            self.server = None
        print("[DebugPanel] Stopped")


def start_debug_panel(port: int = 8081) -> DebugPanelServer:
    """
    Convenience function to start a debug panel server.

    Returns the server instance for later stopping.
    """
    server = DebugPanelServer(port=port)
    server.start()
    return server


# Standalone mode
if __name__ == "__main__":
    import argparse
    import webbrowser

    parser = argparse.ArgumentParser(description="Brain Debug Panel Server")
    parser.add_argument(
        "--port",
        type=int,
        default=8081,
        help="Port to run the debug panel on (default: 8081)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open browser automatically",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Development mode: disable static asset caching",
    )
    args = parser.parse_args()

    if args.dev:
        # In dev mode, always reload assets
        _html_cache = None
        print(
            "[DebugPanel] Development mode: static assets will reload on each request"
        )
        # Monkey-patch to disable caching
        original_get_html = get_debug_panel_html

        def get_debug_panel_html_dev():
            reload_static_assets()
            return original_get_html()

        get_debug_panel_html = get_debug_panel_html_dev

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                  Brain Debug Panel Server                     ║
╠══════════════════════════════════════════════════════════════╣
║  URL: http://localhost:{args.port:<5}                                ║
║                                                              ║
║  This panel shows real-time state of all Brain instances.   ║
║  Make sure to import and use register_brain_for_debug()     ║
║  in your application to see brain states here.              ║
║                                                              ║
║  Endpoints:                                                  ║
║    /           - Debug panel UI                              ║
║    /api/debug  - JSON API for brain states                   ║
║    /reload     - Reload static assets (dev)                  ║
╚══════════════════════════════════════════════════════════════╝
""")

    server = DebugPanelServer(port=args.port)
    server.start()

    if not args.no_browser:
        webbrowser.open(f"http://localhost:{args.port}")

    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[DebugPanel] Shutting down...")
        server.stop()
