from __future__ import annotations

import argparse
import base64
import gc
import io
import json
import mimetypes
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


IMAGE_EXTS = (".jpg", ".jpeg", ".png")
DEFAULT_MODEL_ID = "facebook/sam3"


def _to_numpy(value):
    if hasattr(value, "detach"):
        if value.dtype == torch.bfloat16:
            value = value.float()
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _find_bpe_path() -> Path:
    candidates = [
        Path(__file__).resolve().parent / "sam3" / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz",
        Path("/root/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("could not find SAM3 bpe_simple_vocab_16e6.txt.gz")


def _frame_paths(frames_dir: Path) -> list[Path]:
    return sorted(p for p in frames_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def _png_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{data}"


def _mask_png_b64(mask: np.ndarray) -> str:
    image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _box_list(box) -> list[float]:
    return [float(v) for v in np.asarray(box).reshape(-1)[:4].tolist()]


def _cuda_memory_stats() -> dict:
    if not torch.cuda.is_available():
        return {}
    device = torch.cuda.current_device()
    scale = 1024 * 1024
    return {
        "allocated_mb": round(torch.cuda.memory_allocated(device) / scale, 1),
        "reserved_mb": round(torch.cuda.memory_reserved(device) / scale, 1),
        "max_allocated_mb": round(torch.cuda.max_memory_allocated(device) / scale, 1),
        "max_reserved_mb": round(torch.cuda.max_memory_reserved(device) / scale, 1),
    }


def _cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _detection_from_mask(
    mask: np.ndarray,
    box,
    score: float,
    include_mask: bool,
) -> dict:
    mask = np.squeeze(mask) > 0
    box_list = _box_list(box)
    x0, y0, x1, y1 = box_list
    detection = {
        "score": float(score),
        "box_xyxy": box_list,
        "center_xy": [(x0 + x1) / 2.0, (y0 + y1) / 2.0],
        "area_px": int(mask.sum()),
    }
    if include_mask:
        detection["mask_png_b64"] = _mask_png_b64(mask)
    return detection


class Sam3Session:
    def __init__(self, frames_dir: Path, backend: str = "auto", model_id: str = DEFAULT_MODEL_ID):
        self.frames_dir = frames_dir
        self.frames = _frame_paths(frames_dir)
        if not self.frames:
            raise SystemExit(f"no frames found in {frames_dir}")
        first = Image.open(self.frames[0])
        self.size = first.size
        first.close()
        if backend not in {"auto", "native", "transformers"}:
            raise ValueError("backend must be auto, native, or transformers")
        self.backend = backend
        self.model_id = model_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._lock = threading.Lock()
        self._active_backend = None
        self._processor = None
        self._model = None

    def _processor_ready(self):
        if self._processor is not None:
            return self._processor

        if self.backend in {"auto", "native"}:
            try:
                from sam3.model.sam3_image_processor import Sam3Processor
                from sam3.model_builder import build_sam3_image_model

                try:
                    model = build_sam3_image_model()
                except TypeError:
                    model = build_sam3_image_model(bpe_path=str(_find_bpe_path()))
                self._processor = Sam3Processor(model)
                self._active_backend = "native"
                return self._processor
            except (FileNotFoundError, ModuleNotFoundError):
                if self.backend == "native":
                    raise

        if self.backend in {"auto", "transformers"}:
            from transformers import AutoModel, AutoProcessor

            self._processor = AutoProcessor.from_pretrained(self.model_id)
            self._model = AutoModel.from_pretrained(self.model_id).to(self.device).eval()
            self._active_backend = "transformers"
            return self._processor

        return self._processor

    def _prompt_native(
        self,
        image: Image.Image,
        prompt: str,
        max_masks: int,
        include_masks: bool,
        min_score: float | None,
    ) -> tuple[list[dict], int]:
        processor = self._processor_ready()
        state = processor.set_image(image)
        output = processor.set_text_prompt(state=state, prompt=prompt)
        masks = _to_numpy(output["masks"])
        boxes = _to_numpy(output["boxes"])
        scores = _to_numpy(output["scores"])
        detections = []
        count = min(max_masks, len(masks))
        for mask_idx in range(count):
            score = float(scores[mask_idx])
            if min_score is not None and score < min_score:
                continue
            detections.append(
                _detection_from_mask(
                    masks[mask_idx],
                    boxes[mask_idx],
                    score,
                    include_mask=include_masks,
                )
            )
        return detections, int(len(masks))

    def _prompt_transformers(
        self,
        image: Image.Image,
        prompt: str,
        max_masks: int,
        include_masks: bool,
        min_score: float | None,
    ) -> tuple[list[dict], int]:
        processor = self._processor_ready()
        assert self._model is not None
        if hasattr(processor, "init_video_session") and hasattr(processor, "postprocess_outputs"):
            dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
            session = processor.init_video_session(
                inference_device=self.device,
                inference_state_device=self.device,
                processing_device=self.device,
                video_storage_device=self.device,
                dtype=dtype,
            )
            session = processor.add_text_prompt(session, prompt)
            inputs = processor(images=image, return_tensors="pt").to(self.device)
            outputs = self._model(session, frame=inputs["pixel_values"][0])
            results = processor.postprocess_outputs(
                session,
                outputs,
                original_sizes=inputs.get("original_sizes"),
            )
            masks = _to_numpy(results.get("masks", []))
            boxes = _to_numpy(results.get("boxes", []))
            scores = _to_numpy(results.get("scores", []))
            detections = []
            count = min(max_masks, len(masks))
            for mask_idx in range(count):
                score = float(scores[mask_idx])
                if min_score is not None and score < min_score:
                    continue
                detections.append(
                    _detection_from_mask(
                        masks[mask_idx],
                        boxes[mask_idx],
                        score,
                        include_mask=include_masks,
                    )
                )
            return detections, int(len(masks))

        inputs = processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        outputs = self._model(**inputs)
        target_sizes = inputs.get("original_sizes")
        if hasattr(target_sizes, "tolist"):
            target_sizes = target_sizes.tolist()
        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=0.0 if min_score is None else float(min_score),
            mask_threshold=0.5,
            target_sizes=target_sizes,
        )[0]
        masks = _to_numpy(results.get("masks", []))
        boxes = _to_numpy(results.get("boxes", []))
        scores = _to_numpy(results.get("scores", []))
        detections = []
        count = min(max_masks, len(masks))
        for mask_idx in range(count):
            score = float(scores[mask_idx])
            if min_score is not None and score < min_score:
                continue
            detections.append(
                _detection_from_mask(
                    masks[mask_idx],
                    boxes[mask_idx],
                    score,
                    include_mask=include_masks,
                )
            )
        return detections, int(len(masks))

    def _prompt(
        self,
        image: Image.Image,
        prompt: str,
        max_masks: int,
        include_masks: bool,
        min_score: float | None,
    ) -> tuple[list[dict], int]:
        processor = self._processor_ready()
        del processor
        if self._active_backend == "transformers":
            return self._prompt_transformers(image, prompt, max_masks, include_masks, min_score)
        return self._prompt_native(image, prompt, max_masks, include_masks, min_score)

    def detect(self, frame_idx: int, prompts: list[str], max_masks: int, alpha: float) -> dict:
        if frame_idx < 0 or frame_idx >= len(self.frames):
            raise ValueError(f"frame_idx outside 0..{len(self.frames) - 1}")
        image = Image.open(self.frames[frame_idx]).convert("RGB")
        return self._detect_image(
            image=image,
            frame_label=self.frames[frame_idx].name,
            frame_idx=frame_idx,
            prompts=prompts,
            max_masks=max_masks,
            alpha=alpha,
        )

    def detect_uploaded(
        self,
        image: Image.Image,
        prompts: list[str],
        max_masks: int,
        alpha: float,
        min_score: float,
    ) -> dict:
        return self._detect_image(
            image=image.convert("RGB"),
            frame_label="uploaded",
            frame_idx=None,
            prompts=prompts,
            max_masks=max_masks,
            alpha=alpha,
            include_masks=True,
            min_score=min_score,
        )

    def _detect_image(
        self,
        image: Image.Image,
        frame_label: str,
        frame_idx: int | None,
        prompts: list[str],
        max_masks: int,
        alpha: float,
        include_masks: bool = False,
        min_score: float | None = None,
    ) -> dict:
        prompts = [p.strip() for p in prompts if p.strip()]
        if not prompts:
            raise ValueError("at least one prompt is required")

        base = np.array(image).astype(np.float32)
        overlay = base.copy()
        draw_items = []
        results = []
        top_mask = None
        colors = np.array(
            [
                [32, 170, 255],
                [255, 190, 0],
                [64, 255, 120],
                [255, 70, 120],
                [180, 110, 255],
            ],
            dtype=np.float32,
        )

        memory_before = _cuda_memory_stats()
        with self._lock, torch.inference_mode():
            try:
                autocast = (
                    torch.autocast("cuda", dtype=torch.bfloat16)
                    if torch.cuda.is_available()
                    else torch.autocast("cpu", enabled=False)
                )
                with autocast:
                    self._processor_ready()
                    for prompt_idx, prompt in enumerate(prompts):
                        color = colors[prompt_idx % len(colors)]
                        prompt_results, num_masks = self._prompt(
                            image,
                            prompt,
                            max_masks=max_masks,
                            include_masks=True,
                            min_score=min_score,
                        )
                        response_results = []
                        for mask_idx, detection in enumerate(prompt_results):
                            mask_payload = detection.get("mask_png_b64")
                            if mask_payload:
                                mask_bytes = base64.b64decode(mask_payload)
                                mask_image = Image.open(io.BytesIO(mask_bytes)).convert("L")
                                mask = np.array(mask_image) > 0
                            else:
                                mask = np.zeros(base.shape[:2], dtype=bool)
                            box = detection["box_xyxy"]
                            score = float(detection["score"])
                            overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * color
                            response_detection = dict(detection)
                            if not include_masks:
                                response_detection.pop("mask_png_b64", None)
                            if top_mask is None or score > float(top_mask.get("score", -1.0)):
                                top_mask = {"prompt": prompt, **response_detection}
                            if mask_idx == 0:
                                draw_items.append((prompt, score, box, tuple(int(v) for v in color)))
                            response_results.append(response_detection)
                        results.append(
                            {
                                "prompt": prompt,
                                "num_masks": num_masks,
                                "detections": response_results,
                            }
                        )
            finally:
                _cleanup_cuda()

        out = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
        draw = ImageDraw.Draw(out)
        font = ImageFont.load_default()
        for prompt, score, box, color in draw_items:
            x0, y0, x1, y1 = box
            draw.rectangle((x0, y0, x1, y1), outline=color, width=3)
            label = f"{prompt} {score:.2f}"
            left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
            text_w = right - left
            text_h = bottom - top
            label_y = max(0, y0 - text_h - 6)
            draw.rectangle((x0, label_y, x0 + text_w + 8, label_y + text_h + 6), fill=(0, 0, 0))
            draw.text((x0 + 4, label_y + 3), label, fill=(255, 255, 255), font=font)

        return {
            "frame_idx": frame_idx,
            "frame": frame_label,
            "overlay": _png_data_url(out),
            "results": results,
            "top_mask": top_mask,
            "min_score": min_score,
            "cuda_memory_before": memory_before,
            "cuda_memory_after": _cuda_memory_stats(),
        }


HTML = r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SAM3 Prompt UI</title>
  <style>
    :root { color-scheme: dark; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    body { margin: 0; background: #111; color: #eee; }
    main { display: grid; grid-template-columns: minmax(0, 1fr) 360px; gap: 16px; padding: 16px; min-height: 100vh; box-sizing: border-box; }
    .stage { display: grid; align-content: start; gap: 10px; }
    .viewer { position: relative; width: min(100%, 960px); aspect-ratio: 4 / 3; background: #050505; border: 1px solid #333; overflow: hidden; }
    .viewer img { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: contain; }
    .viewer .empty { position: absolute; inset: 0; display: grid; place-items: center; color: #888; }
    .controls, aside { background: #1a1a1a; border: 1px solid #333; border-radius: 8px; padding: 12px; }
    .controls { width: min(100%, 936px); display: grid; grid-template-columns: auto 1fr auto auto; gap: 10px; align-items: center; }
    button { background: #2b6cff; color: white; border: 0; border-radius: 6px; padding: 8px 10px; font-weight: 650; cursor: pointer; }
    button.secondary { background: #333; }
    button:disabled { opacity: .55; cursor: default; }
    input[type=range] { width: 100%; }
    aside { display: grid; gap: 12px; align-content: start; }
    textarea { width: 100%; height: 116px; box-sizing: border-box; border-radius: 6px; border: 1px solid #444; background: #090909; color: #eee; padding: 8px; resize: vertical; }
    label { display: grid; gap: 6px; font-size: 13px; color: #bbb; }
    .row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
    .chip { border: 1px solid #444; background: #222; color: #eee; border-radius: 999px; padding: 6px 9px; font-size: 12px; }
    .status { color: #aaa; font-size: 13px; min-height: 18px; }
    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    th, td { text-align: left; padding: 6px; border-bottom: 1px solid #333; vertical-align: top; }
    th { color: #bbb; font-weight: 650; }
    @media (max-width: 900px) { main { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
<main>
  <section class="stage">
    <div class="viewer">
      <img id="frame" alt="">
      <img id="overlay" alt="">
      <div id="empty" class="empty">Loading frames...</div>
    </div>
    <div class="controls">
      <button id="play" class="secondary">Play</button>
      <input id="slider" type="range" min="0" max="0" value="0">
      <span id="counter">0 / 0</span>
      <button id="detect">Detect</button>
    </div>
  </section>
  <aside>
    <label>
      Prompts
      <textarea id="prompts">inside of cardboard cylinder
cardboard cylinder
cardboard cylindrical container
cylinder rim</textarea>
    </label>
    <div class="row">
      <button class="chip" data-prompt="inside of cardboard cylinder">inside</button>
      <button class="chip" data-prompt="cardboard cylinder">cardboard cylinder</button>
      <button class="chip" data-prompt="cardboard cylindrical container">cylindrical container</button>
      <button class="chip" data-prompt="cylinder rim">rim</button>
      <button class="chip" data-prompt="brown cardboard container">brown container</button>
    </div>
    <label>
      Max masks per prompt
      <input id="maxMasks" type="range" min="1" max="5" value="1">
    </label>
    <label class="row">
      <input id="autoDetect" type="checkbox">
      Detect while playing
    </label>
    <div id="status" class="status"></div>
    <table>
      <thead><tr><th>Prompt</th><th>Masks</th><th>Top</th><th>Box</th></tr></thead>
      <tbody id="results"></tbody>
    </table>
  </aside>
</main>
<script>
let frameCount = 0;
let playing = false;
let timer = null;
let busy = false;
let lastAutoDetectMs = 0;
const AUTO_DETECT_MIN_MS = 1800;
const AUTO_DETECT_FRAME_STRIDE = 5;
const frame = document.getElementById('frame');
const overlay = document.getElementById('overlay');
const empty = document.getElementById('empty');
const slider = document.getElementById('slider');
const counter = document.getElementById('counter');
const play = document.getElementById('play');
const detectBtn = document.getElementById('detect');
const prompts = document.getElementById('prompts');
const maxMasks = document.getElementById('maxMasks');
const autoDetect = document.getElementById('autoDetect');
const status = document.getElementById('status');
const results = document.getElementById('results');

function promptList() {
  return prompts.value.split(/\n|;/).map(s => s.trim()).filter(Boolean);
}

function setFrame(idx) {
  slider.value = idx;
  frame.src = `/frame?idx=${idx}`;
  overlay.removeAttribute('src');
  counter.textContent = `${Number(idx) + 1} / ${frameCount}`;
}

async function detect() {
  if (busy) return;
  const body = {
    frame_idx: Number(slider.value),
    prompts: promptList(),
    max_masks: Number(maxMasks.value)
  };
  busy = true;
  detectBtn.disabled = true;
  status.textContent = 'Running SAM3...';
  try {
    const resp = await fetch('/api/detect', {
      method: 'POST',
      headers: {'content-type': 'application/json'},
      body: JSON.stringify(body)
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || resp.statusText);
    overlay.src = data.overlay;
    renderResults(data.results);
    status.textContent = `${data.frame}: ${data.results.length} prompt(s)`;
  } catch (err) {
    status.textContent = err.message;
  } finally {
    busy = false;
    detectBtn.disabled = false;
  }
}

function renderResults(items) {
  results.innerHTML = '';
  for (const item of items) {
    const top = item.detections[0];
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${escapeHtml(item.prompt)}</td><td>${item.num_masks}</td><td>${top ? top.score.toFixed(3) : '-'}</td><td>${top ? top.box_xyxy.map(v => Math.round(v)).join(', ') : '-'}</td>`;
    results.appendChild(tr);
  }
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

function tick() {
  let next = Number(slider.value) + 1;
  if (next >= frameCount) next = 0;
  setFrame(next);
  const now = Date.now();
  if (
    autoDetect.checked &&
    !busy &&
    now - lastAutoDetectMs >= AUTO_DETECT_MIN_MS &&
    next % AUTO_DETECT_FRAME_STRIDE === 0
  ) {
    lastAutoDetectMs = now;
    detect();
  }
}

play.onclick = () => {
  playing = !playing;
  play.textContent = playing ? 'Pause' : 'Play';
  if (playing) timer = setInterval(tick, 220);
  else clearInterval(timer);
};
detectBtn.onclick = detect;
slider.oninput = () => setFrame(slider.value);
document.querySelectorAll('[data-prompt]').forEach(btn => {
  btn.onclick = () => {
    const lines = promptList();
    lines.unshift(btn.dataset.prompt);
    prompts.value = [...new Set(lines)].join('\n');
  };
});

fetch('/api/frames').then(r => r.json()).then(data => {
  frameCount = data.frames;
  slider.max = Math.max(0, frameCount - 1);
  empty.style.display = 'none';
  setFrame(0);
});
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    session: Sam3Session

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: dict) -> None:
        self._send(status, json.dumps(payload).encode(), "application/json")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send(200, HTML.encode(), "text/html; charset=utf-8")
            return
        if parsed.path == "/health":
            self._json(
                200,
                {
                    "ok": True,
                    "frames": len(self.session.frames),
                    "width": self.session.size[0],
                    "height": self.session.size[1],
                    "device": str(self.session.device),
                    "backend": self.session.backend,
                    "active_backend": self.session._active_backend,
                    "model_id": self.session.model_id,
                    "cuda_memory": _cuda_memory_stats(),
                },
            )
            return
        if parsed.path == "/api/frames":
            self._json(
                200,
                {
                    "frames": len(self.session.frames),
                    "width": self.session.size[0],
                    "height": self.session.size[1],
                },
            )
            return
        if parsed.path == "/frame":
            q = parse_qs(parsed.query)
            idx = int(q.get("idx", ["0"])[0])
            if idx < 0 or idx >= len(self.session.frames):
                self._json(404, {"error": "frame index out of range"})
                return
            path = self.session.frames[idx]
            data = path.read_bytes()
            self._send(200, data, mimetypes.guess_type(path.name)[0] or "image/jpeg")
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path not in ("/api/detect", "/api/detect_image"):
            self._json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode())
            prompts = payload.get("prompts")
            if isinstance(prompts, str):
                prompts = [prompts]
            if parsed.path == "/api/detect":
                result = self.session.detect(
                    frame_idx=int(payload.get("frame_idx", 0)),
                    prompts=prompts or [],
                    max_masks=max(1, int(payload.get("max_masks", 1))),
                    alpha=float(payload.get("alpha", 0.65)),
                )
            else:
                image_b64 = payload.get("image_b64") or payload.get("image_jpeg_b64") or payload.get("image")
                if not image_b64:
                    raise ValueError("image_b64 is required")
                if "," in image_b64:
                    image_b64 = image_b64.split(",", 1)[1]
                image = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
                result = self.session.detect_uploaded(
                    image=image,
                    prompts=prompts or ["inside of cardboard cylinder"],
                    max_masks=max(1, int(payload.get("max_masks", 1))),
                    alpha=float(payload.get("alpha", 0.65)),
                    min_score=float(payload.get("min_score", 0.25)),
                )
            self._json(200, result)
        except Exception as exc:
            _cleanup_cuda()
            print(f"[sam3-ui] request error: {exc}", flush=True)
            self._json(
                400,
                {
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "cuda_memory": _cuda_memory_stats(),
                },
            )

    def log_message(self, fmt, *args):
        print(f"[sam3-ui] {self.address_string()} {fmt % args}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8202)
    parser.add_argument("--backend", choices=("auto", "native", "transformers"), default="auto")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    args = parser.parse_args()

    Handler.session = Sam3Session(Path(args.frames_dir), backend=args.backend, model_id=args.model_id)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[sam3-ui] serving http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
