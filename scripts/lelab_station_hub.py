#!/usr/bin/env python3
"""Central station hub for LeLab-style fleet control.

Run this on the operator machine next to LeLab. It presents one local API that
LeLab can call, while each Jetson keeps robot IO, safety, recording, and policy
execution local.
"""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from blupe_evals.station import StationConfig, load_station_configs

DEFAULT_TIMEOUT_S = 3.0


HUB_DASHBOARD_HTML = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LeLab Station Dashboard</title>
<style>
:root { color-scheme: dark; font-family: system-ui, -apple-system, Segoe UI, sans-serif; }
body { margin:0; background:#101214; color:#f1f3f4; }
header { display:flex; align-items:center; justify-content:space-between; gap:16px; padding:12px 16px; border-bottom:1px solid #2d3338; background:#15181b; }
h1 { margin:0; font-size:18px; font-weight:650; }
main { padding:14px; display:grid; gap:12px; }
button, input { font:inherit; color:#f1f3f4; background:#1b2025; border:1px solid #3a424a; border-radius:5px; padding:7px 9px; }
button { cursor:pointer; }
button.primary { background:#1f6feb; border-color:#2f81f7; }
button.danger { background:#7f1d1d; border-color:#b91c1c; }
button:disabled { opacity:.5; cursor:not-allowed; }
.row { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
.grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(360px, 1fr)); gap:12px; }
.station { border:1px solid #30363d; border-radius:6px; background:#171b1f; min-width:0; }
.station header { background:transparent; border-bottom:1px solid #30363d; padding:10px 12px; }
.station h2 { margin:0; font-size:16px; }
.body { padding:12px; display:grid; gap:12px; }
.pill { display:inline-flex; align-items:center; gap:6px; border:1px solid #3a424a; border-radius:999px; padding:4px 8px; color:#c9d1d9; font-size:12px; }
.dot { width:8px; height:8px; border-radius:50%; background:#6b7280; }
.ok .dot { background:#22c55e; }
.bad .dot { background:#ef4444; }
.warn .dot { background:#f59e0b; }
.stats { display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:8px; }
.stat { background:#0f1215; border:1px solid #30363d; border-radius:5px; padding:8px; min-height:36px; }
.stat span { display:block; color:#9aa4af; font-size:11px; margin-bottom:3px; }
.stat b { display:block; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; font-size:13px; }
.cams { display:grid; grid-template-columns:repeat(auto-fit, minmax(190px, 1fr)); gap:8px; }
.cam { display:grid; gap:5px; }
.cam img { width:100%; aspect-ratio:4 / 3; object-fit:contain; background:#050607; border:1px solid #30363d; border-radius:5px; }
.cam span { color:#9aa4af; font-size:12px; }
.log { white-space:pre-wrap; max-height:90px; overflow:auto; background:#0f1215; border:1px solid #30363d; border-radius:5px; padding:8px; color:#c9d1d9; font-size:12px; }
.error { color:#ffb4b4; }
.mono { font-family:ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
@media (max-width:700px) {
  .grid { grid-template-columns:1fr; }
  .stats { grid-template-columns:1fr; }
}
</style>
</head>
<body>
<header>
  <h1>LeLab Station Dashboard</h1>
  <div class="row">
    <span class="pill mono" id="hubState"><span class="dot"></span> hub</span>
    <button onclick="refreshAll()">Refresh</button>
  </div>
</header>
<main>
  <section class="grid" id="stations"></section>
</main>
<script>
const state = { stations: [], status: {}, health: {}, errors: {}, leases: {} };

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[c]));
}
function jsString(value) {
  return JSON.stringify(String(value ?? ''));
}
async function getJson(path) {
  const res = await fetch(path, {cache:'no-store'});
  const json = await res.json();
  if (!res.ok) throw new Error(json.error || res.statusText);
  return json;
}
async function postJson(path, payload = {}) {
  const res = await fetch(path, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
  const json = await res.json();
  if (!res.ok) throw new Error(json.error || res.statusText);
  return json;
}
function stationClass(id) {
  if (state.errors[id]) return 'bad';
  const h = state.health[id] || {};
  if (h.ok === false || h.error) return 'bad';
  return state.status[id] ? 'ok' : 'warn';
}
function stationMode(id) {
  const s = state.status[id] || {};
  return s.mode || s.state || s.status || '-';
}
function recordingText(id) {
  const r = (state.status[id] || {}).recording || {};
  if (r.running) return `recording ${r.elapsed || 0}s`;
  return r.dir ? 'stopped' : '-';
}
function teleopText(id) {
  const t = (state.status[id] || {}).teleop || {};
  if (t.claimed || t.active || state.leases[id]) return t.operator || 'claimed';
  return '-';
}
function evalText(id) {
  const e = (state.status[id] || {}).eval || {};
  if (e.running) return `${e.state || 'running'} attempt ${e.attempt || 0}`;
  return e.state || '-';
}
function render() {
  document.getElementById('hubState').className = `pill mono ${state.stations.length ? 'ok' : 'warn'}`;
  document.getElementById('stations').innerHTML = state.stations.map(station => {
    const cls = stationClass(station.id);
    const stationArg = jsString(station.id);
    const cameras = (station.cameras || []).map(camera => `
      <div class="cam">
        <img src="/api/stations/${encodeURIComponent(station.id)}/camera/${encodeURIComponent(camera)}.mjpg" alt="${escapeHtml(camera)} stream">
        <span class="mono">${escapeHtml(camera)}</span>
      </div>
    `).join('');
    const err = state.errors[station.id] ? `<div class="error mono">${escapeHtml(state.errors[station.id])}</div>` : '';
    return `
      <article class="station">
        <header>
          <h2>${escapeHtml(station.name || station.id)}</h2>
          <span class="pill mono ${cls}"><span class="dot"></span>${escapeHtml(station.id)}</span>
        </header>
        <div class="body">
          <div class="stats mono">
            <div class="stat"><span>robot</span><b>${escapeHtml(station.robot_type || '-')}</b></div>
            <div class="stat"><span>mode</span><b>${escapeHtml(stationMode(station.id))}</b></div>
            <div class="stat"><span>recording</span><b>${escapeHtml(recordingText(station.id))}</b></div>
            <div class="stat"><span>teleop</span><b>${escapeHtml(teleopText(station.id))}</b></div>
            <div class="stat"><span>eval</span><b>${escapeHtml(evalText(station.id))}</b></div>
            <div class="stat"><span>base</span><b title="${escapeHtml(station.base_url)}">${escapeHtml(station.base_url || '-')}</b></div>
          </div>
          <div class="cams">${cameras}</div>
          <div class="row">
            <button class="primary" onclick="recordStart(${stationArg})">Record</button>
            <button onclick="recordStop(${stationArg})">Stop Record</button>
            <button onclick="teleopClaim(${stationArg})">Claim Teleop</button>
            <button onclick="teleopRelease(${stationArg})">Release</button>
          </div>
          ${err}
        </div>
      </article>
    `;
  }).join('');
}
async function refreshStation(station) {
  try {
    const [status, health] = await Promise.all([
      getJson(`/api/stations/${encodeURIComponent(station.id)}/status`),
      getJson(`/api/stations/${encodeURIComponent(station.id)}/health`),
    ]);
    state.status[station.id] = status;
    state.health[station.id] = health;
    state.errors[station.id] = '';
  } catch (err) {
    state.errors[station.id] = err.message;
  }
}
async function refreshAll() {
  try {
    const data = await getJson('/api/stations');
    state.stations = data.stations || [];
    await Promise.all(state.stations.map(refreshStation));
  } catch (err) {
    document.getElementById('stations').innerHTML = `<div class="error mono">${escapeHtml(err.message)}</div>`;
    return;
  }
  render();
}
async function recordStart(id) {
  await postJson(`/api/stations/${encodeURIComponent(id)}/record/start`, {capture_mode:'continuous', cameras:[]});
  await refreshAll();
}
async function recordStop(id) {
  await postJson(`/api/stations/${encodeURIComponent(id)}/record/stop`, {});
  await refreshAll();
}
async function teleopClaim(id) {
  const data = await postJson(`/api/stations/${encodeURIComponent(id)}/teleop/claim`, {operator:'lelab'});
  state.leases[id] = data?.teleop?.lease_id || data?.lease_id || '';
  await refreshAll();
}
async function teleopRelease(id) {
  const lease_id = state.leases[id] || (state.status[id]?.teleop || {}).lease_id || '';
  await postJson(`/api/stations/${encodeURIComponent(id)}/teleop/release`, {lease_id, outcome:'complete'});
  state.leases[id] = '';
  await refreshAll();
}
refreshAll();
setInterval(refreshAll, 3000);
</script>
</body>
</html>
"""


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Headers", "content-type")
    handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _html_response(handler: BaseHTTPRequestHandler, body_text: str) -> None:
    body = body_text.encode("utf-8")
    handler.send_response(200)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _station_by_id(stations: dict[str, StationConfig], station_id: str) -> StationConfig:
    try:
        return stations[station_id]
    except KeyError as exc:
        raise ValueError(f"unknown station: {station_id}") from exc


def _station_get_json(station: StationConfig, path: str, timeout_s: float = DEFAULT_TIMEOUT_S) -> dict[str, Any]:
    resp = requests.get(f"{station.normalized_base_url}{path}", timeout=timeout_s)
    resp.raise_for_status()
    data = resp.json()
    return data if isinstance(data, dict) else {"response": data}


def _station_post_json(
    station: StationConfig,
    path: str,
    payload: dict[str, Any],
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    resp = requests.post(f"{station.normalized_base_url}{path}", json=payload, timeout=timeout_s)
    resp.raise_for_status()
    data = resp.json()
    return data if isinstance(data, dict) else {"response": data}


def _proxy_camera(handler: BaseHTTPRequestHandler, station: StationConfig, camera: str, suffix: str) -> None:
    if camera not in station.cameras:
        _json_response(handler, 404, {"error": f"camera {camera!r} is not configured for station {station.id}"})
        return
    upstream = f"{station.normalized_base_url}/camera/{camera}.{suffix}"
    try:
        with requests.get(upstream, stream=True, timeout=(3.0, 30.0)) as resp:
            resp.raise_for_status()
            handler.send_response(200)
            handler.send_header("Content-Type", resp.headers.get("Content-Type", "image/jpeg"))
            handler.send_header("Cache-Control", "no-store")
            handler.send_header("Access-Control-Allow-Origin", "*")
            handler.end_headers()
            for chunk in resp.iter_content(chunk_size=16384):
                if not chunk:
                    continue
                handler.wfile.write(chunk)
                handler.wfile.flush()
    except (BrokenPipeError, ConnectionResetError):
        return
    except Exception as exc:
        try:
            _json_response(handler, 502, {"error": f"camera proxy failed: {exc}", "upstream": upstream})
        except (BrokenPipeError, ConnectionResetError):
            return


def make_handler(stations_list: list[StationConfig]):
    stations = {station.id: station for station in stations_list}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def do_OPTIONS(self) -> None:
            _json_response(self, 204, {})

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            parts = [part for part in parsed.path.split("/") if part]
            try:
                if parsed.path in {"/", "/dashboard"}:
                    _html_response(self, HUB_DASHBOARD_HTML)
                    return
                if parsed.path == "/api/stations":
                    _json_response(self, 200, {"stations": [station.as_public_dict() for station in stations_list]})
                    return
                if len(parts) >= 3 and parts[0] == "api" and parts[1] == "stations":
                    station = _station_by_id(stations, parts[2])
                    if len(parts) == 3:
                        _json_response(self, 200, {"station": station.as_public_dict()})
                        return
                    if parts[3] == "status":
                        _json_response(self, 200, _station_get_json(station, "/api/status"))
                        return
                    if parts[3] == "health":
                        _json_response(self, 200, _station_get_json(station, "/api/health"))
                        return
                    if len(parts) == 5 and parts[3] == "camera":
                        camera_file = parts[4]
                        if "." not in camera_file:
                            raise ValueError("camera route must end in .jpg or .mjpg")
                        camera, suffix = camera_file.rsplit(".", 1)
                        if suffix not in {"jpg", "mjpg"}:
                            raise ValueError("camera route must end in .jpg or .mjpg")
                        _proxy_camera(self, station, camera, suffix)
                        return
                _json_response(self, 404, {"error": "not found"})
            except ValueError as exc:
                _json_response(self, 404, {"error": str(exc)})
            except Exception as exc:
                _json_response(self, 502, {"error": str(exc)})

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            parts = [part for part in parsed.path.split("/") if part]
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                _json_response(self, 400, {"error": "invalid json"})
                return
            if not isinstance(payload, dict):
                _json_response(self, 400, {"error": "payload must be a JSON object"})
                return

            try:
                if len(parts) >= 4 and parts[0] == "api" and parts[1] == "stations":
                    station = _station_by_id(stations, parts[2])
                    command = "/".join(parts[3:])
                    allowed = {
                        "record/start": "/api/record/start",
                        "record/stop": "/api/record/stop",
                        "teleop/claim": "/api/teleop/claim",
                        "teleop/heartbeat": "/api/teleop/heartbeat",
                        "teleop/release": "/api/teleop/release",
                        "eval/start": "/api/eval/start",
                        "eval/stop": "/api/eval/stop",
                        "eval/resume": "/api/eval/resume",
                        "eval/clear": "/api/eval/clear",
                    }
                    if command not in allowed:
                        _json_response(self, 404, {"error": f"unsupported station command: {command}"})
                        return
                    _json_response(self, 200, _station_post_json(station, allowed[command], payload))
                    return
                _json_response(self, 404, {"error": "not found"})
            except ValueError as exc:
                _json_response(self, 404, {"error": str(exc)})
            except Exception as exc:
                _json_response(self, 502, {"error": str(exc)})

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument("--stations", required=True, help="JSON file with station configs.")
    args = parser.parse_args()

    stations = load_station_configs(Path(args.stations))
    server = ThreadingHTTPServer((args.host, args.port), make_handler(stations))
    print(
        f"LeLab station hub listening on http://{args.host}:{args.port} "
        f"for {len(stations)} station(s)",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
