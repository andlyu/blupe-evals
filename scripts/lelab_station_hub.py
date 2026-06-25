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
from urllib.parse import parse_qs, quote, urlparse

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
            <button class="primary" onclick="location.href='/record?station=${encodeURIComponent(station.id)}'">Record Dataset</button>
            <button onclick="recordStart(${stationArg})">Quick Record</button>
            <button onclick="recordStop(${stationArg})">Stop Record</button>
            <button onclick="teleopClaim(${stationArg})">Claim Teleop</button>
            <button onclick="teleopRelease(${stationArg})">Release</button>
            <button onclick="location.href='/dataset?station=${encodeURIComponent(station.id)}'">Edit Dataset</button>
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


DATASET_EDITOR_HTML = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LeLab Dataset Editor</title>
<style>
:root { color-scheme: dark; font-family: system-ui, -apple-system, Segoe UI, sans-serif; }
body { margin:0; background:#101214; color:#f1f3f4; }
header { display:flex; align-items:center; justify-content:space-between; gap:16px; padding:12px 16px; border-bottom:1px solid #2d3338; background:#15181b; }
h1 { margin:0; font-size:18px; }
a { color:#8ab4f8; }
main { padding:14px; display:grid; gap:12px; }
.panel { border:1px solid #30363d; border-radius:6px; background:#171b1f; padding:12px; }
.row { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
button, input, select, textarea { font:inherit; color:#f1f3f4; background:#1b2025; border:1px solid #3a424a; border-radius:5px; padding:7px 9px; }
button { cursor:pointer; }
button.primary { background:#1f6feb; border-color:#2f81f7; }
button:disabled { opacity:.5; cursor:not-allowed; }
.stats { display:grid; grid-template-columns:repeat(auto-fit, minmax(140px, 1fr)); gap:8px; }
.stat { background:#0f1215; border:1px solid #30363d; border-radius:5px; padding:8px; min-height:36px; }
.stat span { display:block; color:#9aa4af; font-size:11px; margin-bottom:3px; }
.stat b { display:block; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; font-size:13px; }
.cams { display:grid; grid-template-columns:repeat(auto-fit, minmax(260px, 1fr)); gap:10px; }
.cam { display:grid; gap:5px; }
.cam img { width:100%; aspect-ratio:4 / 3; object-fit:contain; background:#050607; border:1px solid #30363d; border-radius:5px; }
.cam span { color:#9aa4af; font-size:12px; }
table { width:100%; border-collapse:collapse; font-size:13px; }
th, td { border-bottom:1px solid #30363d; padding:6px; text-align:left; vertical-align:top; }
th { color:#9aa4af; font-weight:600; }
td input, td select, td textarea { width:100%; box-sizing:border-box; }
td textarea { min-height:38px; }
.compact { width:74px; }
.status { color:#9aa4af; min-height:20px; }
.error { color:#ffb4b4; }
.mono { font-family:ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
#frameSlider { flex:1 1 360px; min-width:180px; }
@media (max-width:700px) {
  .stats { grid-template-columns:1fr; }
  .cams { grid-template-columns:1fr; }
}
</style>
</head>
<body>
<header>
  <h1>LeLab Dataset Editor</h1>
  <a href="/dashboard">Station dashboard</a>
</header>
<main>
  <section class="panel">
    <div class="row">
      <label>Station <select id="stationSelect"></select></label>
      <label>Recording <select id="recordingSelect"></select></label>
      <button onclick="loadRecordings()">Refresh</button>
      <button onclick="stepFrame(-1)">Prev</button>
      <button id="playButton" class="primary" onclick="togglePlay()">Play</button>
      <button onclick="stepFrame(1)">Next</button>
      <button onclick="markStart()">Mark Start</button>
      <button onclick="markEnd()">Mark End</button>
      <label>Frame <input id="frameNumber" type="number" min="0" value="0" style="width:84px"></label>
      <input id="frameSlider" type="range" min="0" max="0" value="0">
    </div>
  </section>
  <section class="stats mono" id="stats"></section>
  <section class="cams" id="cams"></section>
  <section class="panel">
    <div class="row" style="justify-content:space-between">
      <div class="row">
        <button onclick="addSegment()">Add Segment</button>
        <button onclick="loadManifest()">Load Manifest</button>
        <button onclick="saveManifest()">Save Manifest</button>
        <button class="primary" onclick="exportSegments()">Export Episodes</button>
      </div>
      <div class="status mono" id="manifestStatus"></div>
    </div>
    <div style="overflow:auto; margin-top:10px">
      <table>
        <thead>
          <tr><th>Start</th><th>End</th><th>Task</th><th>Outcome</th><th>Type</th><th>Notes</th><th></th></tr>
        </thead>
        <tbody id="segmentsBody"></tbody>
      </table>
    </div>
  </section>
  <section class="panel">
    <div class="mono" id="sample"></div>
  </section>
</main>
<script>
let stations = [];
let recordings = [];
let recording = null;
let frameIdx = 0;
let playTimer = null;
let segments = [];

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[c]));
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
function selectedStation() { return document.getElementById('stationSelect').value; }
function selectedRecording() { return document.getElementById('recordingSelect').value; }
function queryStation() { return new URLSearchParams(location.search).get('station') || ''; }
function cameraList() {
  const cams = Array.isArray(recording?.cameras) ? recording.cameras : [];
  if (cams.length) return cams.map(cam => ({name: cam.name || cam, frames_dir: cam.frames_dir || cam.name || cam}));
  return Object.keys(recording?.frames || {}).map(name => ({name, frames_dir: name}));
}
function timestampFor(idx) {
  const sample = (recording?.samples || [])[idx] || {};
  if (sample.timestamp_s !== undefined) return Number(sample.timestamp_s);
  const fps = Math.max(1, Number(recording?.meta?.fps || 30));
  return idx / fps;
}
function frameFor(camera, idx) {
  const frames = recording?.frames?.[camera] || [];
  if (!frames.length) return null;
  return frames[Math.max(0, Math.min(idx, frames.length - 1))];
}
async function loadStations() {
  const data = await getJson('/api/stations');
  stations = data.stations || [];
  const select = document.getElementById('stationSelect');
  const selected = select.value || queryStation() || stations[0]?.id || '';
  select.innerHTML = stations.map(st => `<option value="${escapeHtml(st.id)}">${escapeHtml(st.name || st.id)}</option>`).join('');
  if (selected && stations.some(st => st.id === selected)) select.value = selected;
  if (select.value) await loadRecordings();
}
async function loadRecordings() {
  const station = selectedStation();
  const data = await getJson(`/api/stations/${encodeURIComponent(station)}/recordings`);
  recordings = data.episodes || data.recordings || [];
  const select = document.getElementById('recordingSelect');
  const selected = select.value || recordings[0]?.name || '';
  select.innerHTML = recordings.map(ep => `<option value="${escapeHtml(ep.name)}">${escapeHtml(ep.name)} (${escapeHtml(ep.kind || 'recording')})</option>`).join('');
  if (selected && recordings.some(ep => ep.name === selected)) select.value = selected;
  if (select.value) await loadRecording(select.value);
}
async function loadRecording(name) {
  const station = selectedStation();
  recording = await getJson(`/api/stations/${encodeURIComponent(station)}/recording?name=${encodeURIComponent(name)}`);
  frameIdx = 0;
  segments = (recording.manifest?.segments || []).map(seg => ({...seg}));
  const maxIdx = Math.max(0, Number(recording.length || 0) - 1);
  document.getElementById('frameSlider').max = maxIdx;
  document.getElementById('frameNumber').max = maxIdx;
  renderCameras();
  renderSegments();
  renderFrame(0);
}
function renderCameras() {
  document.getElementById('cams').innerHTML = cameraList().map(cam => `
    <div class="cam">
      <img id="camera-${escapeHtml(cam.name)}" alt="${escapeHtml(cam.name)} frame">
      <span class="mono" id="camera-${escapeHtml(cam.name)}-info"></span>
    </div>
  `).join('');
}
function renderFrame(idx) {
  if (!recording) return;
  frameIdx = Math.max(0, Math.min(idx, Math.max(0, Number(recording.length || 0) - 1)));
  document.getElementById('frameSlider').value = frameIdx;
  document.getElementById('frameNumber').value = frameIdx;
  const station = selectedStation();
  for (const cam of cameraList()) {
    const frame = frameFor(cam.name, frameIdx);
    const img = document.getElementById(`camera-${cam.name}`);
    const info = document.getElementById(`camera-${cam.name}-info`);
    if (!img || !info) continue;
    if (frame) {
      img.src = `/api/stations/${encodeURIComponent(station)}/frame?name=${encodeURIComponent(recording.name)}&camera=${encodeURIComponent(cam.name)}&frame=${encodeURIComponent(frame.frame)}`;
      info.textContent = `${cam.name} ${frame.frame} t=${frame.timestamp_s ?? '-'}s`;
    } else {
      img.removeAttribute('src');
      info.textContent = `${cam.name} no frame`;
    }
  }
  const sample = (recording.samples || [])[frameIdx] || {};
  document.getElementById('sample').innerHTML = [
    `<div>sample ${frameIdx} / ${Math.max(0, Number(recording.length || 0) - 1)}</div>`,
    `<div>timestamp ${sample.timestamp_s ?? '-'}</div>`,
    `<div>state ${escapeHtml(Array.isArray(sample.observation_state) ? sample.observation_state.map(v => Number(v).toFixed(2)).join(', ') : '-')}</div>`,
    `<div>action ${escapeHtml(Array.isArray(sample.action) ? sample.action.map(v => Number(v).toFixed(2)).join(', ') : '-')}</div>`,
  ].join('');
  renderStats();
}
function renderStats() {
  const meta = recording?.meta || {};
  const counts = recording?.counts || {};
  const cameraCounts = cameraList().map(cam => `${cam.name}:${counts[cam.name] || 0}`).join(' ');
  document.getElementById('stats').innerHTML = [
    `<div class="stat"><span>task</span><b title="${escapeHtml(meta.task || '')}">${escapeHtml(meta.task || '-')}</b></div>`,
    `<div class="stat"><span>kind</span><b>${escapeHtml(recording?.kind || '-')}</b></div>`,
    `<div class="stat"><span>samples</span><b>${counts.samples || 0}</b></div>`,
    `<div class="stat"><span>cameras</span><b title="${escapeHtml(cameraCounts)}">${escapeHtml(cameraCounts || '-')}</b></div>`,
    `<div class="stat"><span>fps</span><b>${Number(meta.fps || 0).toFixed(1)}</b></div>`,
  ].join('');
}
function renderSegments() {
  document.getElementById('segmentsBody').innerHTML = segments.map((seg, idx) => `
    <tr>
      <td><input class="compact" type="number" step="0.001" value="${Number(seg.start_s || 0)}" onchange="updateSegment(${idx}, 'start_s', this.value)"></td>
      <td><input class="compact" type="number" step="0.001" value="${Number(seg.end_s || 0)}" onchange="updateSegment(${idx}, 'end_s', this.value)"></td>
      <td><input value="${escapeHtml(seg.task || '')}" onchange="updateSegment(${idx}, 'task', this.value)"></td>
      <td><select onchange="updateSegment(${idx}, 'outcome', this.value)">
        <option value="success" ${seg.outcome === 'success' ? 'selected' : ''}>success</option>
        <option value="failure" ${seg.outcome === 'failure' ? 'selected' : ''}>failure</option>
      </select></td>
      <td><select onchange="updateSegment(${idx}, 'type', this.value)">
        <option value="teleop" ${seg.type !== 'intervention' ? 'selected' : ''}>teleop</option>
        <option value="intervention" ${seg.type === 'intervention' ? 'selected' : ''}>intervention</option>
      </select></td>
      <td><textarea onchange="updateSegment(${idx}, 'notes', this.value)">${escapeHtml(seg.notes || '')}</textarea></td>
      <td><button onclick="removeSegment(${idx})">Remove</button></td>
    </tr>
  `).join('');
}
function updateSegment(idx, key, value) {
  if (!segments[idx]) return;
  segments[idx][key] = ['start_s', 'end_s'].includes(key) ? Number(value) : value;
}
function addSegment() {
  const t = timestampFor(frameIdx);
  segments.push({start_s:Number(t.toFixed(3)), end_s:Number((t + 5).toFixed(3)), task:recording?.meta?.task || '', outcome:'success', type:'teleop', notes:''});
  renderSegments();
}
function removeSegment(idx) { segments.splice(idx, 1); renderSegments(); }
function markStart() {
  if (!segments.length) addSegment();
  segments[segments.length - 1].start_s = Number(timestampFor(frameIdx).toFixed(3));
  if (segments[segments.length - 1].end_s <= segments[segments.length - 1].start_s) {
    segments[segments.length - 1].end_s = Number((segments[segments.length - 1].start_s + 1).toFixed(3));
  }
  renderSegments();
}
function markEnd() {
  if (!segments.length) addSegment();
  segments[segments.length - 1].end_s = Number(timestampFor(frameIdx).toFixed(3));
  renderSegments();
}
async function loadManifest() {
  const data = await getJson(`/api/stations/${encodeURIComponent(selectedStation())}/segments?source=${encodeURIComponent(selectedRecording())}`);
  segments = (data.segments || []).map(seg => ({...seg}));
  document.getElementById('manifestStatus').textContent = `loaded ${segments.length} segment(s)`;
  renderSegments();
}
async function saveManifest() {
  const data = await postJson(`/api/stations/${encodeURIComponent(selectedStation())}/segments/save`, {source:selectedRecording(), segments});
  segments = (data.segments || []).map(seg => ({...seg}));
  document.getElementById('manifestStatus').textContent = `saved ${segments.length} segment(s)`;
  renderSegments();
}
async function exportSegments() {
  const data = await postJson(`/api/stations/${encodeURIComponent(selectedStation())}/segments/export`, {source:selectedRecording(), segments});
  document.getElementById('manifestStatus').textContent = `created ${data.episode_count || 0} episode(s), ${data.total_frames || 0} frames`;
  await loadRecordings();
}
function stepFrame(delta) { renderFrame(frameIdx + delta); }
function togglePlay() {
  if (playTimer) {
    clearInterval(playTimer);
    playTimer = null;
    document.getElementById('playButton').textContent = 'Play';
    return;
  }
  document.getElementById('playButton').textContent = 'Pause';
  const fps = Math.max(1, Math.min(60, Number(recording?.meta?.fps || 30)));
  playTimer = setInterval(() => {
    if (!recording || frameIdx >= Number(recording.length || 0) - 1) {
      togglePlay();
      return;
    }
    renderFrame(frameIdx + 1);
  }, Math.max(16, Math.round(1000 / fps)));
}
document.getElementById('stationSelect').addEventListener('change', loadRecordings);
document.getElementById('recordingSelect').addEventListener('change', e => loadRecording(e.target.value));
document.getElementById('frameSlider').addEventListener('input', e => renderFrame(Number(e.target.value)));
document.getElementById('frameNumber').addEventListener('change', e => renderFrame(Number(e.target.value)));
loadStations().catch(err => {
  document.getElementById('stats').innerHTML = `<div class="stat error"><span>error</span><b>${escapeHtml(err.message)}</b></div>`;
});
</script>
</body>
</html>
"""


RECORD_DATASET_HTML = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LeLab Station Recording</title>
<style>
:root { color-scheme: dark; font-family: system-ui, -apple-system, Segoe UI, sans-serif; }
body { margin:0; background:#101214; color:#f1f3f4; }
header { display:flex; align-items:center; justify-content:space-between; gap:16px; padding:12px 16px; border-bottom:1px solid #2d3338; background:#15181b; }
h1 { margin:0; font-size:18px; }
a { color:#8ab4f8; }
main { padding:14px; display:grid; gap:12px; grid-template-columns:minmax(360px, 520px) minmax(360px, 1fr); align-items:start; }
.panel { border:1px solid #30363d; border-radius:6px; background:#171b1f; padding:12px; display:grid; gap:12px; }
.full { grid-column:1 / -1; }
.row { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
.field { display:grid; gap:4px; }
.grid { display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:10px; }
label, th { color:#9aa4af; font-size:12px; }
button, input, select, textarea { font:inherit; color:#f1f3f4; background:#1b2025; border:1px solid #3a424a; border-radius:5px; padding:7px 9px; }
textarea { min-height:68px; resize:vertical; }
button { cursor:pointer; }
button.primary { background:#1f6feb; border-color:#2f81f7; }
button.danger { background:#7f1d1d; border-color:#b91c1c; }
button:disabled { opacity:.5; cursor:not-allowed; }
table { width:100%; border-collapse:collapse; font-size:13px; }
th, td { border-bottom:1px solid #30363d; padding:7px 6px; text-align:left; vertical-align:middle; }
td input { width:100%; box-sizing:border-box; }
.cams { display:grid; grid-template-columns:repeat(auto-fit, minmax(220px, 1fr)); gap:10px; }
.cam { display:grid; gap:5px; }
.cam img { width:100%; aspect-ratio:4 / 3; object-fit:contain; background:#050607; border:1px solid #30363d; border-radius:5px; }
.cam span { color:#9aa4af; font-size:12px; }
.status { min-height:22px; color:#9aa4af; }
.error { color:#ffb4b4; }
.ok { color:#b6f3c5; }
.mono { font-family:ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
@media (max-width:900px) {
  main { grid-template-columns:1fr; }
  .grid { grid-template-columns:1fr; }
}
</style>
</head>
<body>
<header>
  <h1>LeLab Station Recording</h1>
  <div class="row">
    <a href="/dashboard">Station dashboard</a>
    <a href="/dataset">Dataset editor</a>
  </div>
</header>
<main>
  <section class="panel">
    <div class="field">
      <label for="stationSelect">Station</label>
      <select id="stationSelect"></select>
    </div>
    <div class="field">
      <label for="robotProfileSelect">Station robot</label>
      <select id="robotProfileSelect"></select>
    </div>
    <div class="grid">
      <div class="field">
        <label for="datasetRepoId">Dataset repo</label>
        <input id="datasetRepoId" value="andlyu/so101-blue-ball">
      </div>
      <div class="field">
        <label for="captureMode">Capture mode</label>
        <select id="captureMode">
          <option value="continuous" selected>continuous</option>
          <option value="policy_execute">policy execute</option>
        </select>
      </div>
      <div class="field">
        <label for="numEpisodes">Episodes</label>
        <input id="numEpisodes" type="number" min="1" value="5">
      </div>
      <div class="field">
        <label for="episodeTimeS">Episode time</label>
        <input id="episodeTimeS" type="number" min="1" value="30">
      </div>
      <div class="field">
        <label for="resetTimeS">Reset time</label>
        <input id="resetTimeS" type="number" min="0" value="10">
      </div>
      <div class="field">
        <label for="fps">FPS</label>
        <input id="fps" type="number" min="1" value="30">
      </div>
    </div>
    <div class="field">
      <label for="task">Task</label>
      <textarea id="task">Pick up the blue ball and place it in the cup</textarea>
    </div>
    <div class="row">
      <label><input id="video" type="checkbox" checked> video</label>
      <label><input id="pushToHub" type="checkbox"> push to Hub</label>
      <label><input id="privateDataset" type="checkbox"> private</label>
    </div>
    <div class="row">
      <button class="primary" onclick="startRecording()">Start Recording</button>
      <button class="danger" onclick="stopRecording()">Stop</button>
      <button onclick="loadPreset()">Refresh Station</button>
    </div>
    <div id="formStatus" class="status mono"></div>
  </section>
  <section class="panel">
    <div class="row" style="justify-content:space-between">
      <h2 style="margin:0; font-size:16px">Station Cameras</h2>
      <button onclick="addCameraRow()">Add Camera</button>
    </div>
    <div style="overflow:auto">
      <table>
        <thead><tr><th>Role</th><th>Type</th><th>ID</th><th>Port</th><th>Connected</th></tr></thead>
        <tbody id="robotRows"></tbody>
      </table>
    </div>
    <div style="overflow:auto">
      <table>
        <thead><tr><th>Use</th><th>Name</th><th>Source</th><th>FPS</th></tr></thead>
        <tbody id="cameraRows"></tbody>
      </table>
    </div>
    <div class="cams" id="cameraPreview"></div>
  </section>
  <section class="panel full">
    <div class="row" style="justify-content:space-between">
      <h2 style="margin:0; font-size:16px">Payload Preview</h2>
      <button onclick="copyPayload()">Copy JSON</button>
    </div>
    <pre id="payloadPreview" class="mono" style="white-space:pre-wrap; margin:0"></pre>
  </section>
</main>
<script>
let stations = [];
let preset = null;
let cameras = [];

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[c]));
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
function queryStation() { return new URLSearchParams(location.search).get('station') || ''; }
function selectedStation() { return document.getElementById('stationSelect').value; }
function value(id) { return document.getElementById(id).value; }
function numberValue(id) { return Number(value(id)); }
function checked(id) { return document.getElementById(id).checked; }
function setStatus(text, cls = '') {
  const el = document.getElementById('formStatus');
  el.className = `status mono ${cls}`;
  el.textContent = text;
}
async function loadStations() {
  const data = await getJson('/api/stations');
  stations = data.stations || [];
  const select = document.getElementById('stationSelect');
  const selected = select.value || queryStation() || stations[0]?.id || '';
  select.innerHTML = stations.map(st => `<option value="${escapeHtml(st.id)}">${escapeHtml(st.name || st.id)}</option>`).join('');
  if (selected && stations.some(st => st.id === selected)) select.value = selected;
  if (select.value) await loadPreset();
}
async function loadPreset() {
  const station = selectedStation();
  preset = await getJson(`/api/stations/${encodeURIComponent(station)}/recording-preset`);
  renderRobotProfiles();
  cameras = (preset.cameras || []).map(cam => ({...cam, enabled: cam.enabled !== false, fps: cam.fps || numberValue('fps') || 30}));
  document.getElementById('fps').value = preset.defaults?.fps || document.getElementById('fps').value || 30;
  renderRobots();
  renderCameras();
  renderPayload();
  setStatus(`loaded ${cameras.length} camera preset(s) from ${station}`, 'ok');
}
function robotProfiles() {
  const profiles = preset?.robot_profiles || [];
  if (profiles.length) return profiles;
  const robots = preset?.robots || [];
  const follower = robots.find(robot => robot.role === 'follower') || null;
  const leader = robots.find(robot => robot.role === 'leader') || null;
  return [{
    id: 'station_robot',
    name: 'Station robot',
    robot_type: preset?.station?.robot_type || 'so101',
    follower,
    leader,
    cameras: (preset?.cameras || []).map(cam => cam.name),
  }];
}
function selectedRobotProfile() {
  const id = document.getElementById('robotProfileSelect')?.value || '';
  return robotProfiles().find(profile => profile.id === id) || robotProfiles()[0] || null;
}
function renderRobotProfiles() {
  const select = document.getElementById('robotProfileSelect');
  const profiles = robotProfiles();
  const selected = select.value || profiles[0]?.id || '';
  select.innerHTML = profiles.map(profile => `<option value="${escapeHtml(profile.id)}">${escapeHtml(profile.name || profile.id)}</option>`).join('');
  if (selected && profiles.some(profile => profile.id === selected)) select.value = selected;
}
function renderRobots() {
  const profile = selectedRobotProfile();
  const robots = [profile?.follower, profile?.leader].filter(Boolean);
  document.getElementById('robotRows').innerHTML = robots.map(robot => `
    <tr>
      <td class="mono">${escapeHtml(robot.role || '-')}</td>
      <td class="mono">${escapeHtml(robot.type || '-')}</td>
      <td class="mono">${escapeHtml(robot.id || '-')}</td>
      <td class="mono">${escapeHtml(robot.port || '-')}</td>
      <td class="mono">${robot.connected === null || robot.connected === undefined ? '-' : escapeHtml(robot.connected)}</td>
    </tr>
  `).join('');
}
function renderCameras() {
  document.getElementById('cameraRows').innerHTML = cameras.map((cam, idx) => `
    <tr>
      <td><input type="checkbox" ${cam.enabled ? 'checked' : ''} onchange="cameras[${idx}].enabled = this.checked; renderPayload()"></td>
      <td><input value="${escapeHtml(cam.name)}" onchange="cameras[${idx}].name = this.value; renderPayload()"></td>
      <td><input value="${escapeHtml(cam.stream_url || cam.snapshot_url || '')}" onchange="cameras[${idx}].stream_url = this.value; renderPayload()"></td>
      <td><input type="number" min="1" value="${Number(cam.fps || 15)}" onchange="cameras[${idx}].fps = Number(this.value); renderPayload()"></td>
    </tr>
  `).join('');
  document.getElementById('cameraPreview').innerHTML = cameras.filter(cam => cam.enabled).map(cam => `
    <div class="cam">
      <img src="${escapeHtml(cam.stream_url || cam.snapshot_url)}" alt="${escapeHtml(cam.name)} stream">
      <span class="mono">${escapeHtml(cam.name)} ${escapeHtml(cam.lerobot_key || '')}</span>
    </div>
  `).join('');
}
function addCameraRow() {
  cameras.push({type:'station', station_id:selectedStation(), name:'camera', enabled:true, fps:numberValue('fps') || 30, stream_url:'', snapshot_url:''});
  renderCameras();
  renderPayload();
}
function payload() {
  const selectedCameras = cameras.filter(cam => cam.enabled);
  const profile = selectedRobotProfile();
  const episodeTime = Math.max(1, numberValue('episodeTimeS') || 30);
  const resetTime = Math.max(0, numberValue('resetTimeS') || 0);
  const episodes = Math.max(1, numberValue('numEpisodes') || 1);
  const duration = episodes * episodeTime + Math.max(0, episodes - 1) * resetTime;
  const datasetRepoId = value('datasetRepoId').trim();
  const namePrefix = datasetRepoId.split('/').pop() || 'so101_recording';
  return {
    station_id: selectedStation(),
    robot_profile_id: profile?.id || '',
    robot_profile: profile,
    dataset_repo_id: datasetRepoId,
    single_task: value('task').trim(),
    num_episodes: episodes,
    episode_time_s: episodeTime,
    reset_time_s: resetTime,
    fps: Math.max(1, numberValue('fps') || 30),
    video: checked('video'),
    push_to_hub: checked('pushToHub'),
    private: checked('privateDataset'),
    capture_mode: value('captureMode'),
    duration_s: duration,
    name_prefix: namePrefix,
    cameras: selectedCameras.map(cam => cam.name),
    camera_configs: Object.fromEntries(selectedCameras.map(cam => [cam.name, {
      type: 'station',
      station_id: selectedStation(),
      stream_url: cam.stream_url,
      snapshot_url: cam.snapshot_url,
      lerobot_key: cam.lerobot_key || `observation.images.${cam.name}`,
    }])),
  };
}
function renderPayload() {
  document.getElementById('payloadPreview').textContent = JSON.stringify(payload(), null, 2);
}
async function startRecording() {
  const body = payload();
  const station = selectedStation();
  setStatus('starting recording...');
  const result = await postJson(`/api/stations/${encodeURIComponent(station)}/record/start`, {
    duration_s: body.duration_s,
    fps: body.fps,
    cameras: body.cameras,
    task: body.single_task,
    name_prefix: body.name_prefix,
    capture_mode: body.capture_mode,
    extra_meta: body,
  });
  setStatus(`recording started on station: ${result.dir || ''}`, 'ok');
}
async function stopRecording() {
  const station = selectedStation();
  await postJson(`/api/stations/${encodeURIComponent(station)}/record/stop`, {});
  setStatus('recording stop requested', 'ok');
}
async function copyPayload() {
  await navigator.clipboard.writeText(JSON.stringify(payload(), null, 2));
  setStatus('payload copied', 'ok');
}
for (const id of ['datasetRepoId', 'task', 'numEpisodes', 'episodeTimeS', 'resetTimeS', 'fps', 'video', 'pushToHub', 'privateDataset', 'captureMode']) {
  document.addEventListener('input', event => {
    if (event.target && event.target.id === id) renderPayload();
  });
  document.addEventListener('change', event => {
    if (event.target && event.target.id === id) renderPayload();
  });
}
document.getElementById('stationSelect').addEventListener('change', loadPreset);
document.getElementById('robotProfileSelect').addEventListener('change', () => { renderRobots(); renderPayload(); });
loadStations().catch(err => setStatus(err.message, 'error'));
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


def _station_recording_preset(station: StationConfig) -> dict[str, Any]:
    status: dict[str, Any] = {}
    try:
        status = _station_get_json(station, "/api/status")
    except Exception as exc:
        status = {"error": str(exc)}
    status_cameras = {
        str(camera.get("name")): camera
        for camera in status.get("cameras", [])
        if isinstance(camera, dict) and camera.get("name")
    }
    cameras: list[dict[str, Any]] = []
    for camera_name in station.cameras:
        upstream = status_cameras.get(camera_name, {})
        cameras.append(
            {
                "type": "station",
                "station_id": station.id,
                "name": camera_name,
                "enabled": True,
                "stream_url": f"/api/stations/{quote(station.id, safe='')}/camera/{quote(camera_name, safe='')}.mjpg",
                "snapshot_url": f"/api/stations/{quote(station.id, safe='')}/camera/{quote(camera_name, safe='')}.jpg",
                "source_url": upstream.get("url", ""),
                "frames_dir": upstream.get("frames_dir", camera_name),
                "frames_file": upstream.get("frames_file", f"{camera_name}/frames.jsonl"),
                "lerobot_key": upstream.get("lerobot_key", f"observation.images.{camera_name}"),
                "fps": 30,
            }
        )
    return {
        "station": station.as_public_dict(),
        "status": status,
        "robots": status.get("robots", []),
        "robot_profiles": status.get("robot_profiles", []),
        "defaults": {
            "num_episodes": 5,
            "episode_time_s": 30,
            "reset_time_s": 10,
            "fps": 30,
            "video": True,
            "capture_mode": "continuous",
        },
        "cameras": cameras,
    }


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


def _proxy_station_file(handler: BaseHTTPRequestHandler, station: StationConfig, path: str) -> None:
    upstream = f"{station.normalized_base_url}{path}"
    try:
        with requests.get(upstream, stream=True, timeout=(3.0, 30.0)) as resp:
            resp.raise_for_status()
            handler.send_response(200)
            handler.send_header("Content-Type", resp.headers.get("Content-Type", "application/octet-stream"))
            handler.send_header("Cache-Control", resp.headers.get("Cache-Control", "no-store"))
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
            _json_response(handler, 502, {"error": f"file proxy failed: {exc}", "upstream": upstream})
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
                if parsed.path in {"/record", "/record-dataset"}:
                    _html_response(self, RECORD_DATASET_HTML)
                    return
                if parsed.path == "/dataset":
                    _html_response(self, DATASET_EDITOR_HTML)
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
                    if parts[3] == "recording-preset":
                        _json_response(self, 200, _station_recording_preset(station))
                        return
                    if parts[3] == "recordings":
                        _json_response(self, 200, _station_get_json(station, "/api/episodes"))
                        return
                    if parts[3] == "recording":
                        q = parse_qs(parsed.query)
                        name = q.get("name", ["latest"])[0]
                        _json_response(self, 200, _station_get_json(station, f"/api/episode?name={quote(name, safe='')}"))
                        return
                    if parts[3] == "segments":
                        q = parse_qs(parsed.query)
                        source = q.get("source", ["latest"])[0]
                        _json_response(self, 200, _station_get_json(station, f"/api/segments?source={quote(source, safe='')}"))
                        return
                    if parts[3] == "frame":
                        q = parse_qs(parsed.query)
                        name = q.get("name", ["latest"])[0]
                        camera = q.get("camera", [""])[0]
                        frame = q.get("frame", [""])[0]
                        _proxy_station_file(
                            self,
                            station,
                            f"/episode_frame?name={quote(name, safe='')}&camera={quote(camera, safe='')}&frame={quote(frame, safe='')}",
                        )
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
                        "segments/save": "/api/segments/save",
                        "segments/export": "/api/busyboard/extract",
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
