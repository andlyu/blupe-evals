"""Judge recorded eval trials and render the report — companion to mac_quest_bridge.py --task.

Trials live in runs/<session>/trial_NNN/ (video.mp4 + meta.json), written by TrialRecorder.

  serve   local judging UI (stdlib http.server): play each video, click success/fail,
          on fail pick the stage it failed at + type a 0-1 score for how far it got,
          notes optional. Judgments save straight into each trial's meta.json.
  render  write a self-contained report.html into the session dir: success rate, mean
          progress score, failure-by-stage histogram, per-trial rows with embedded video.

Scoring: success = 1.0; fail = (completed stages + typed 0-1) / n stages (just the typed
score if the task has no stages). Comparable across sessions of the same task.

  python scripts/eval_report.py serve            # newest session under runs/
  python scripts/eval_report.py render --session runs/2026-06-11_red-plate-pickup
"""

import json
import os
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

import tyro


def latest_session(runs_dir="runs"):
    dirs = [os.path.join(runs_dir, d) for d in sorted(os.listdir(runs_dir))
            if os.path.isdir(os.path.join(runs_dir, d))] if os.path.isdir(runs_dir) else []
    if not dirs:
        sys.exit(f"no sessions under {runs_dir}/ — run mac_quest_bridge.py --task <name> first")
    return dirs[-1]


def load_session(session):
    task = {"task": os.path.basename(session), "stages": []}
    tj = os.path.join(session, "task.json")
    if os.path.exists(tj):
        task = json.load(open(tj))
    trials = []
    for d in sorted(os.listdir(session)):
        mj = os.path.join(session, d, "meta.json")
        if d.startswith("trial_") and os.path.exists(mj):
            trials.append({"name": d, "meta": json.load(open(mj)),
                           "video": os.path.exists(os.path.join(session, d, "video.mp4"))})
    return task, trials


def trial_score(meta, stages):
    """success=1.0; fail=(completed stages + 0-1 at the failed stage)/n; None if unjudged."""
    if meta.get("result") == "success":
        return 1.0
    if meta.get("result") != "fail" or meta.get("score") is None:
        return None
    s = float(meta["score"])
    if stages and meta.get("failed_stage") in stages:
        return (stages.index(meta["failed_stage"]) + s) / len(stages)
    return s


def summarize(task, trials):
    judged = [t for t in trials if t["meta"].get("result")]
    scores = [trial_score(t["meta"], task["stages"]) for t in judged]
    scores = [s for s in scores if s is not None]
    wins = sum(1 for t in judged if t["meta"]["result"] == "success")
    by_stage = {}
    for t in judged:
        if t["meta"]["result"] == "fail" and t["meta"].get("failed_stage"):
            by_stage[t["meta"]["failed_stage"]] = by_stage.get(t["meta"]["failed_stage"], 0) + 1
    return {"n": len(trials), "judged": len(judged), "success": wins,
            "rate": wins / len(judged) if judged else None,
            "mean_score": sum(scores) / len(scores) if scores else None,
            "failed_by_stage": by_stage}


# ------------------------------------------------------------------- judge UI

JUDGE_PAGE = """<!doctype html><meta charset="utf-8"><title>judge — {task}</title>
<style>
 body{{font:14px -apple-system,sans-serif;background:#16181c;color:#dde;margin:24px}}
 h1{{font-size:18px}} .trial{{background:#1e2128;border-radius:10px;padding:14px;margin:14px 0;
 display:flex;gap:16px;align-items:flex-start}} video{{width:560px;border-radius:6px}}
 .judged{{outline:2px solid #2c6}} .fail{{outline-color:#c44}}
 button{{margin:2px;padding:6px 12px;border:0;border-radius:6px;background:#3a3f4a;color:#dde;cursor:pointer}}
 button.on{{background:#2c6;color:#fff}} button.failon{{background:#c44;color:#fff}}
 input[type=number]{{width:64px}} textarea{{width:100%;background:#14161a;color:#dde;border:1px solid #333}}
 .meta{{color:#89a;font-size:12px}}
</style>
<h1>{task} — judge ({n} trials)</h1><div id=list></div>
<script>
const STAGES = {stages};
async function main() {{
  const data = await (await fetch('/data')).json();
  const list = document.getElementById('list');
  for (const t of data.trials) {{
    const m = t.meta, div = document.createElement('div');
    div.className = 'trial' + (m.result ? (m.result === 'fail' ? ' judged fail' : ' judged') : '');
    div.innerHTML = `
      <div><video controls preload="metadata" src="/file/${{t.name}}/video.mp4"></video>
        <div class=meta>${{t.name}} · ${{m.duration_s ?? '?'}}s · ${{(m.events||[]).map(e=>e[1]+':'+e[2]).join(' ')}}</div></div>
      <div style="flex:1">
        <div><button class="res ${{m.result==='success'?'on':''}}" data-r=success>SUCCESS</button>
             <button class="res ${{m.result==='fail'?'failon':''}}" data-r=fail>FAIL</button></div>
        <div class=stg style="display:${{m.result==='fail'?'block':'none'}}">
          failed at: ${{STAGES.map(s=>`<button class="stage ${{m.failed_stage===s?'failon':''}}" data-s="${{s}}">${{s}}</button>`).join('')}}
          progress in that stage (0-1): <input type=number class=score min=0 max=1 step=0.05 value="${{m.score ?? ''}}">
        </div>
        <textarea class=notes rows=2 placeholder="notes">${{m.notes || ''}}</textarea>
        <div><button class=save>save</button> <span class=st></span></div>
      </div>`;
    const state = {{ result: m.result, failed_stage: m.failed_stage }};
    div.querySelectorAll('.res').forEach(b => b.onclick = () => {{
      state.result = b.dataset.r;
      div.querySelectorAll('.res').forEach(x => x.className = 'res');
      b.className = 'res ' + (state.result === 'fail' ? 'failon' : 'on');
      div.querySelector('.stg').style.display = state.result === 'fail' ? 'block' : 'none';
    }});
    div.querySelectorAll('.stage').forEach(b => b.onclick = () => {{
      state.failed_stage = b.dataset.s;
      div.querySelectorAll('.stage').forEach(x => x.className = 'stage');
      b.className = 'stage failon';
    }});
    div.querySelector('.save').onclick = async () => {{
      const body = {{ trial: t.name, result: state.result, failed_stage: state.failed_stage,
        score: parseFloat(div.querySelector('.score').value),
        notes: div.querySelector('.notes').value }};
      if (!body.result) return div.querySelector('.st').textContent = 'pick success/fail';
      if (body.result === 'fail' && STAGES.length && !body.failed_stage)
        return div.querySelector('.st').textContent = 'pick the failed stage';
      if (body.result === 'fail' && isNaN(body.score))
        return div.querySelector('.st').textContent = 'type the 0-1 score';
      const r = await fetch('/judge', {{method: 'POST', body: JSON.stringify(body)}});
      div.querySelector('.st').textContent = r.ok ? 'saved ✓' : 'ERROR';
      div.className = 'trial judged' + (body.result === 'fail' ? ' fail' : '');
    }};
    list.appendChild(div);
  }}
}}
main();
</script>"""


def make_handler(session):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="text/html"):
            data = body if isinstance(body, bytes) else body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            task, trials = load_session(session)
            if self.path == "/":
                self._send(200, JUDGE_PAGE.format(task=task["task"], n=len(trials),
                                                  stages=json.dumps(task["stages"])))
            elif self.path == "/data":
                self._send(200, json.dumps({"task": task, "trials": trials}),
                           "application/json")
            elif self.path.startswith("/file/"):
                self._video(os.path.join(session, *self.path[6:].split("/")))
            else:
                self._send(404, "not found")

        def _video(self, path):
            """Serve mp4 with Range support (Safari/Chrome require it for <video> seek)."""
            if not (os.path.realpath(path).startswith(os.path.realpath(session))
                    and os.path.exists(path)):
                return self._send(404, "not found")
            size = os.path.getsize(path)
            rng = re.match(r"bytes=(\d+)-(\d*)", self.headers.get("Range", ""))
            start = int(rng.group(1)) if rng else 0
            end = int(rng.group(2)) if rng and rng.group(2) else size - 1
            with open(path, "rb") as f:
                f.seek(start)
                data = f.read(end - start + 1)
            self.send_response(206 if rng else 200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            if rng:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            if self.path != "/judge":
                return self._send(404, "not found")
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            name = body.get("trial", "")
            mj = os.path.join(session, name, "meta.json")
            if not re.fullmatch(r"trial_\d+", name) or not os.path.exists(mj):
                return self._send(400, "bad trial")
            meta = json.load(open(mj))
            meta["result"] = body.get("result")
            meta["failed_stage"] = body.get("failed_stage") if meta["result"] == "fail" else None
            score = body.get("score")
            meta["score"] = max(0.0, min(1.0, float(score))) \
                if meta["result"] == "fail" and score is not None else None
            meta["notes"] = body.get("notes", "")
            with open(mj, "w") as f:
                json.dump(meta, f, indent=2)
            print(f"[judge] {name}: {meta['result']}"
                  + (f" @ {meta['failed_stage']}={meta['score']}" if meta["result"] == "fail" else ""),
                  flush=True)
            self._send(200, "ok", "text/plain")
    return Handler


# ------------------------------------------------------------------- report

REPORT_PAGE = """<!doctype html><meta charset="utf-8"><title>{task} — eval report</title>
<style>
 body{{font:14px -apple-system,sans-serif;background:#fff;color:#222;margin:32px;max-width:1100px}}
 h1{{font-size:22px}} .kpi{{display:inline-block;background:#f4f5f7;border-radius:10px;
 padding:12px 20px;margin-right:12px;text-align:center}} .kpi b{{font-size:22px;display:block}}
 table{{border-collapse:collapse;width:100%;margin-top:20px}}
 td,th{{border-bottom:1px solid #e5e5e5;padding:10px;text-align:left;vertical-align:top}}
 video{{width:420px;border-radius:6px}} .ok{{color:#1a8f4a;font-weight:600}}
 .bad{{color:#c0392b;font-weight:600}} .bar{{background:#c0392b;height:14px;border-radius:3px}}
 .muted{{color:#999}}
</style>
<h1>{task} — eval report <span class=muted>({date})</span></h1>
<div><span class=kpi><b>{n}</b>trials</span><span class=kpi><b>{rate}</b>success rate</span>
<span class=kpi><b>{mean}</b>mean progress</span><span class=kpi><b>{judged}</b>judged</span></div>
{histogram}
<table><tr><th>trial</th><th>video</th><th>result</th><th>score</th><th>events</th><th>notes</th></tr>
{rows}</table>"""


def render(session):
    task, trials = load_session(session)
    s = summarize(task, trials)
    hist = ""
    if s["failed_by_stage"]:
        peak = max(s["failed_by_stage"].values())
        bars = "".join(
            f"<tr><td>{st}</td><td style='width:70%'><div class=bar style='width:{100*c/peak:.0f}%'></div></td><td>{c}</td></tr>"
            for st, c in sorted(s["failed_by_stage"].items(),
                                key=lambda kv: task["stages"].index(kv[0]) if kv[0] in task["stages"] else 99))
        hist = f"<h3>failures by stage</h3><table style='max-width:560px'>{bars}</table>"
    rows = []
    for t in trials:
        m = t["meta"]
        score = trial_score(m, task["stages"])
        res = ("<span class=ok>success</span>" if m.get("result") == "success" else
               f"<span class=bad>fail @ {m.get('failed_stage') or '?'}</span>" if m.get("result") == "fail"
               else "<span class=muted>unjudged</span>")
        ev = " ".join(f"{e[0]}s:{e[1]}={e[2]}" for e in m.get("events", []))
        vid = (f"<video controls preload=metadata src='{t['name']}/video.mp4'></video>"
               if t["video"] else "<span class=muted>no video</span>")
        rows.append(f"<tr><td>{t['name']}<br><span class=muted>{m.get('duration_s', '?')}s</span></td>"
                    f"<td>{vid}</td><td>{res}</td>"
                    f"<td>{'' if score is None else f'{score:.2f}'}</td>"
                    f"<td class=muted style='font-size:12px'>{ev}</td><td>{m.get('notes', '')}</td></tr>")
    html = REPORT_PAGE.format(
        task=task["task"], date=os.path.basename(session).split("_")[0], n=s["n"],
        judged=s["judged"],
        rate="—" if s["rate"] is None else f"{100*s['rate']:.0f}%",
        mean="—" if s["mean_score"] is None else f"{s['mean_score']:.2f}",
        histogram=hist, rows="\n".join(rows))
    out = os.path.join(session, "report.html")
    with open(out, "w") as f:
        f.write(html)
    rate = "—" if s["rate"] is None else f"{100 * s['rate']:.0f}%"
    mean = "—" if s["mean_score"] is None else f"{s['mean_score']:.2f}"
    print(f"[report] {out}")
    print(f"[report] {s['judged']}/{s['n']} judged · success {rate} · mean {mean}")
    return out


def main(mode: tyro.conf.Positional[str], session: Optional[str] = None, port: int = 7799):
    """mode: serve (judging UI) | render (write report.html). session: runs/<dir>,
    default = newest."""
    session = session or latest_session()
    if mode == "render":
        render(session)
    elif mode == "serve":
        srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(session))
        print(f"[judge] {session} -> http://127.0.0.1:{port}/  (Ctrl-C to stop)")
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass
    else:
        sys.exit("mode must be 'serve' or 'render'")


if __name__ == "__main__":
    tyro.cli(main)
