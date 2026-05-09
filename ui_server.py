"""ui_server.py — Flask UI for the autofilm autoresearch pipeline.

Exposes a single web page where the user can:
  1. Configure a run (book PDF path, moodboard examples, optional director,
     optional cinematographer, number of iterations)
  2. Start the autoresearch loop in the background
  3. Watch images and videos populate a grid as each iteration progresses
  4. Run the API smoke test before committing to a costly real run

The pipeline runs as a subprocess (``python run_loop.py``) so the UI is
fully decoupled from the long-running generation; the frontend just polls
``/api/state`` and re-renders.

User-uploaded moodboard images are saved to
``experiments/<book_slug>/user_moodboards/`` (per-book, not per-experiment)
so they're automatically inherited across iterations. produce.py reads
that directory at the lookbook + moodboard stages and uses the images as
style references.

Usage
-----
    pip install flask          # already in deps
    python ui_server.py
    open http://localhost:5174

Env vars (override defaults):
    AUTOFILM_UI_PORT  — listen port (default 5174)
    AUTOFILM_UI_HOST  — bind address (default 127.0.0.1)
"""
from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from flask import (
    Flask,
    abort,
    jsonify,
    request,
    send_file,
    send_from_directory,
)


# ── Project paths ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import after path setup so the UI can stand alone.
from prepare import EXPERIMENTS_DIR, iter_all_experiments  # noqa: E402

UI_DIR = PROJECT_ROOT / "ui"


# ── Run state ────────────────────────────────────────────────────────────────

class RunState:
    """In-memory state for the currently-running pipeline.

    Only one run at a time — starting a new one terminates any existing
    process. Logs are kept in a circular buffer for tail display.
    """
    def __init__(self) -> None:
        self.lock      = threading.Lock()
        self.process: subprocess.Popen | None = None
        self.config:   dict[str, Any] = {}
        self.started_at:   float | None = None
        self.finished_at:  float | None = None
        self.status:       str = "idle"   # idle | running | done | error
        self.exit_code:    int | None = None
        self.log_buffer:   list[str] = []
        self.log_max:      int = 4000

    def append_log(self, line: str) -> None:
        with self.lock:
            self.log_buffer.append(line.rstrip())
            if len(self.log_buffer) > self.log_max:
                self.log_buffer = self.log_buffer[-self.log_max:]

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "status":       self.status,
                "config":       self.config,
                "started_at":   self.started_at,
                "finished_at":  self.finished_at,
                "exit_code":    self.exit_code,
                "is_running":   self.process is not None and self.process.poll() is None,
                "log_tail":     self.log_buffer[-200:],
            }


run_state  = RunState()
smoke_state = RunState()   # separate state for the API smoke test


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_book_slug(book_path: Path) -> str:
    """Match the slug derivation used by prepare._book_slug() so user
    moodboards are written to the same directory the pipeline reads."""
    return "".join(
        c.lower() if c.isalnum() else "_" for c in book_path.stem
    ).strip("_") or "book"


def _spawn_subprocess(
    state: RunState,
    cmd: list[str],
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> None:
    """Start a subprocess in a background thread and tee its output into
    state.log_buffer. Updates state.status accordingly."""
    with state.lock:
        if state.process and state.process.poll() is None:
            try:
                state.process.send_signal(signal.SIGTERM)
            except Exception:
                pass

        state.log_buffer.clear()
        state.status      = "running"
        state.started_at  = time.time()
        state.finished_at = None
        state.exit_code   = None

        full_env = {**os.environ, **(env or {})}
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd or PROJECT_ROOT),
            env=full_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        state.process = proc

    state.append_log(f"$ {' '.join(shlex.quote(c) for c in cmd)}")

    def _reader() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            state.append_log(line)
        proc.wait()
        with state.lock:
            state.exit_code   = proc.returncode
            state.finished_at = time.time()
            state.status      = "done" if proc.returncode == 0 else "error"

    threading.Thread(target=_reader, daemon=True).start()


def _safe_relative_path(root: Path, candidate: str) -> Path:
    """Resolve ``candidate`` relative to ``root``; raise 404 if escapes."""
    candidate = candidate.lstrip("/")
    full = (root / candidate).resolve()
    if not full.is_relative_to(root.resolve()):
        abort(404)
    if not full.exists():
        abort(404)
    return full


# ── Flask app ─────────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder=None)


# ── Frontend routes ──────────────────────────────────────────────────────────

@app.route("/")
def index() -> Any:
    return send_from_directory(UI_DIR, "index.html")


@app.route("/static/<path:fname>")
def static_files(fname: str) -> Any:
    return send_from_directory(UI_DIR / "static", fname)


# ── State endpoints ──────────────────────────────────────────────────────────

@app.get("/api/state")
def api_state() -> Any:
    """Aggregate state for the frontend: run state, smoke-test state, and
    a list of experiments with their production_bible.json contents (so
    the grid has everything it needs in one round trip)."""
    experiments: list[dict[str, Any]] = []
    for exp_path in sorted(iter_all_experiments(), key=lambda p: p.stat().st_mtime, reverse=True):
        item: dict[str, Any] = {
            "exp_id":      exp_path.name,
            "book_slug":   exp_path.parent.name,
            "qualified":   f"{exp_path.parent.name}/{exp_path.name}",
            "mtime":       exp_path.stat().st_mtime,
            "has_final":   (exp_path / "final.mp4").exists(),
            "has_metric":  (exp_path / "metric.json").exists(),
        }
        pb = exp_path / "production_bible.json"
        if pb.exists():
            try:
                item["bible"] = json.loads(pb.read_text())
            except Exception as e:  # noqa: BLE001
                item["bible_error"] = str(e)
        experiments.append(item)

    return jsonify({
        "run":          run_state.snapshot(),
        "smoke":        smoke_state.snapshot(),
        "experiments":  experiments[:30],   # cap; UI shows latest 30
    })


@app.get("/api/log/run")
def api_log_run() -> Any:
    """Full log of the current/last run (not just tail)."""
    with run_state.lock:
        return jsonify({"lines": run_state.log_buffer.copy()})


@app.get("/api/log/smoke")
def api_log_smoke() -> Any:
    with smoke_state.lock:
        return jsonify({"lines": smoke_state.log_buffer.copy()})


# ── Run-loop start ───────────────────────────────────────────────────────────

@app.post("/api/start")
def api_start() -> Any:
    """Spawn the autoresearch loop with form-supplied parameters.

    Form fields (multipart):
      book_path        — absolute path to the book PDF (required)
      iterations       — int (default 3)
      target_loss      — float (default 0.15)
      threshold        — 'low' | 'medium' | 'high' (default 'medium')
      director         — string, optional
      cinematographer  — string, optional
      moodboards       — list of file uploads, optional
    """
    if run_state.snapshot()["is_running"]:
        return jsonify({"error": "A run is already in progress."}), 409

    book_path_str = (request.form.get("book_path") or "").strip()
    if not book_path_str:
        return jsonify({"error": "book_path is required"}), 400
    book_path = Path(book_path_str).expanduser()
    if not book_path.is_file():
        return jsonify({"error": f"book_path does not point to a file: {book_path}"}), 400

    iterations  = max(1, min(int(request.form.get("iterations", "3")), 20))
    target_loss = float(request.form.get("target_loss", "0.15"))
    threshold   = request.form.get("threshold", "medium")
    if threshold not in ("low", "medium", "high"):
        threshold = "medium"

    director         = (request.form.get("director") or "").strip()
    cinematographer  = (request.form.get("cinematographer") or "").strip()

    # Save uploaded moodboards to experiments/<book_slug>/user_moodboards/.
    book_slug = _safe_book_slug(book_path)
    user_mb_dir = EXPERIMENTS_DIR / book_slug / "user_moodboards"
    user_mb_dir.mkdir(parents=True, exist_ok=True)
    saved_moodboards: list[str] = []
    for f in request.files.getlist("moodboards"):
        if not f or not f.filename:
            continue
        name = Path(f.filename).name
        # Only accept image extensions.
        if Path(name).suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            continue
        dest = user_mb_dir / name
        f.save(dest)
        saved_moodboards.append(str(dest.relative_to(EXPERIMENTS_DIR)))

    cmd = [
        sys.executable, "run_loop.py",
        "--iterations", str(iterations),
        "--target",     f"{target_loss}",
        "--threshold",  threshold,
    ]
    env: dict[str, str] = {
        "BOOK_PDF_PATH": str(book_path),
    }
    if director:
        env["DIRECTOR"] = director
    if cinematographer:
        env["CINEMATOGRAPHER"] = cinematographer

    config = {
        "book_path":         str(book_path),
        "book_slug":         book_slug,
        "iterations":        iterations,
        "target_loss":       target_loss,
        "threshold":         threshold,
        "director":          director,
        "cinematographer":   cinematographer,
        "moodboards_saved":  saved_moodboards,
    }
    run_state.config = config
    _spawn_subprocess(run_state, cmd, env=env)

    return jsonify({"ok": True, "config": config})


@app.post("/api/stop")
def api_stop() -> Any:
    """Politely terminate the current run."""
    with run_state.lock:
        if run_state.process and run_state.process.poll() is None:
            try:
                run_state.process.send_signal(signal.SIGTERM)
            except Exception as e:  # noqa: BLE001
                return jsonify({"error": str(e)}), 500
            return jsonify({"ok": True, "stopped": True})
    return jsonify({"ok": True, "stopped": False, "note": "no run in progress"})


# ── API smoke test trigger ────────────────────────────────────────────────────

@app.post("/api/smoke-test")
def api_smoke_test() -> Any:
    """Run scripts/api_smoke_test.py with the given category filter."""
    if smoke_state.snapshot()["is_running"]:
        return jsonify({"error": "Smoke test already running."}), 409

    category = request.form.get("category", "all")
    cmd = [sys.executable, "scripts/api_smoke_test.py"]
    if category == "image":
        cmd.append("--image")
    elif category == "video":
        cmd.append("--video")
    elif category == "audio":
        cmd.append("--audio")
    smoke_state.config = {"category": category}
    _spawn_subprocess(smoke_state, cmd)
    return jsonify({"ok": True, "category": category})


# ── Artifact serving (images, videos) ────────────────────────────────────────

@app.get("/api/artifact/<book_slug>/<exp_id>/<path:relpath>")
def api_artifact(book_slug: str, exp_id: str, relpath: str) -> Any:
    """Serve a single file from an experiment dir. Path must stay inside
    experiments/<book_slug>/<exp_id>/ — leading '..' is rejected."""
    exp_root = EXPERIMENTS_DIR / book_slug / exp_id
    if not exp_root.is_dir():
        abort(404)
    full = _safe_relative_path(exp_root, relpath)
    return send_file(full)


# ── Smoke run for testing the UI plumbing itself ──────────────────────────────

@app.get("/api/health")
def health() -> Any:
    return jsonify({"ok": True, "experiments_dir": str(EXPERIMENTS_DIR)})


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    host = os.environ.get("AUTOFILM_UI_HOST", "127.0.0.1")
    port = int(os.environ.get("AUTOFILM_UI_PORT", "5174"))
    print(f"autofilm UI on http://{host}:{port}")
    print(f"  experiments dir: {EXPERIMENTS_DIR}")
    app.run(host=host, port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
