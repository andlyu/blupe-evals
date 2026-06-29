#!/usr/bin/env python3
"""Probe browser camera access from the LeLab origin.

The LeLab recording UI needs browser `MediaDeviceInfo.deviceId` values for
preview. Backend `/available-cameras` only proves OpenCV/AVFoundation can list
devices. This script installs a temporary probe page into LeLab's static dist,
opens it in Chrome at `http://127.0.0.1:8000`, and waits for the browser to post
what `navigator.mediaDevices` sees.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


PROBE_HTML = """<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>LeLab Camera Probe</title>
    <style>
      body { font-family: system-ui, sans-serif; padding: 24px; background: #111; color: #eee; }
      pre { white-space: pre-wrap; background: #222; padding: 16px; border-radius: 8px; }
    </style>
  </head>
  <body>
    <h1>LeLab Camera Probe</h1>
    <pre id="out">Running...</pre>
    <script>
      const post = async (payload) => {
        document.getElementById("out").textContent = JSON.stringify(payload, null, 2);
        await fetch("http://127.0.0.1:8765/result", {
          method: "POST",
          mode: "cors",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        }).catch(() => {});
      };
      const listDevices = async () => {
        if (!navigator.mediaDevices?.enumerateDevices) return [];
        const devices = await navigator.mediaDevices.enumerateDevices();
        return devices
          .filter((d) => d.kind === "videoinput")
          .map((d, index) => ({
            index,
            kind: d.kind,
            label: d.label,
            deviceId: d.deviceId,
            groupId: d.groupId,
          }));
      };
      const withTimeout = (promise, ms) =>
        Promise.race([
          promise,
          new Promise((_, reject) => setTimeout(() => reject(new Error("timeout")), ms)),
        ]);
      (async () => {
        const payload = {
          href: location.href,
          isSecureContext,
          hasMediaDevices: !!navigator.mediaDevices,
          userAgent: navigator.userAgent,
          before: [],
          after: [],
          getUserMedia: null,
        };
        try {
          payload.before = await listDevices();
        } catch (error) {
          payload.beforeError = { name: error.name, message: error.message };
        }
        try {
          const stream = await withTimeout(
            navigator.mediaDevices.getUserMedia({ video: true }),
            8000
          );
          payload.getUserMedia = {
            ok: true,
            tracks: stream.getVideoTracks().map((track) => ({
              label: track.label,
              readyState: track.readyState,
              settings: track.getSettings ? track.getSettings() : {},
            })),
          };
          stream.getTracks().forEach((track) => track.stop());
        } catch (error) {
          payload.getUserMedia = { ok: false, name: error.name, message: error.message };
        }
        try {
          payload.after = await listDevices();
        } catch (error) {
          payload.afterError = { name: error.name, message: error.message };
        }
        await post(payload);
      })();
    </script>
  </body>
</html>
"""


class ResultHandler(BaseHTTPRequestHandler):
    result: dict | None = None
    event = threading.Event()

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.end_headers()

    def do_POST(self) -> None:
        if self.path != "/result":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            ResultHandler.result = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            ResultHandler.result = {"decode_error": raw.decode("utf-8", errors="replace")}
        ResultHandler.event.set()
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok":true}')

    def log_message(self, format: str, *args) -> None:
        return


def lelab_dist() -> Path:
    try:
        import lelab
    except ImportError as exc:
        raise SystemExit("Could not import lelab. Activate the LeLab Python environment first.") from exc
    return Path(lelab.__file__).resolve().parents[1] / "frontend" / "dist"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000/camera-probe.html")
    parser.add_argument("--timeout-s", type=float, default=18.0)
    parser.add_argument("--open-browser", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    dist = lelab_dist()
    dist.mkdir(parents=True, exist_ok=True)
    probe_path = dist / "camera-probe.html"
    probe_path.write_text(PROBE_HTML)

    server = ThreadingHTTPServer(("127.0.0.1", 8765), ResultHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        if args.open_browser:
            subprocess.run(["open", "-a", "Google Chrome", args.url], check=False)
        deadline = time.time() + args.timeout_s
        while time.time() < deadline:
            if ResultHandler.event.wait(timeout=0.25):
                print(json.dumps(ResultHandler.result, indent=2))
                return 0
        print(json.dumps({"timeout": True, "probe_url": args.url}, indent=2))
        return 2
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
