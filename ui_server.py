"""ui_server.py — Flask UI for the autofilm autoresearch pipeline.

Single web page that lets the user:
  1. Configure a run (book PDF — drag-drop or path, moodboard examples,
     optional director, optional cinematographer, number of iterations)
  2. Start the autoresearch loop in the background
  3. Watch progress (current iteration, current stage) update live
  4. Browse experiments via a sidebar dropdown — selecting one loads its
     defaults into the form so it's easy to re-run with the same settings
  5. Run the API smoke test before committing to a costly real run

The pipeline runs as a subprocess (``python run_loop.py``) so the UI is
fully decoupled from the long-running generation; the frontend polls
``/api/state`` and re-renders.

Each run writes a ``run_config.json`` to the experiment dir capturing the
exact settings used. The dropdown reads these files to populate defaults
when the user selects an experiment.

User-uploaded moodboard images are saved per-book to
``experiments/<book_slug>/user_moodboards/`` so they're inherited across
iterations of the loop. produce.py's lookbook stage reads that directory
and uses the images as style references.

Drag-dropped PDFs are saved to ``~/.autofilm/uploads/`` and the resolved
path is then used as ``BOOK_PDF_PATH``.

Usage
-----
    pip install flask
    python ui_server.py
    open http://localhost:5174

Env vars (override defaults):
    AUTOFILM_UI_PORT  — listen port (default 5174)
    AUTOFILM_UI_HOST  — bind address (default 127.0.0.1)
"""
from __future__ import annotations

import json
import os
import re
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
import prepare  # noqa: E402
from prepare import iter_all_experiments  # noqa: E402

# Allow UI smoke tests to point the server at an isolated temp dir
# without touching the real experiments tree. When the env var is set,
# we override prepare.EXPERIMENTS_DIR — that's the single source of
# truth used by every helper (iter_all_experiments, Experiment.load,
# etc.) so all endpoints behave consistently against the override.
_TEST_EXP_DIR = os.environ.get("AUTOFILM_TEST_EXPERIMENTS_DIR")
if _TEST_EXP_DIR:
    prepare.EXPERIMENTS_DIR = Path(_TEST_EXP_DIR)

EXPERIMENTS_DIR = prepare.EXPERIMENTS_DIR

UI_DIR     = PROJECT_ROOT / "ui"
UPLOAD_DIR = Path.home() / ".autofilm" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Persistent "last run" config — written on successful start, loaded on
# page refresh so the user doesn't have to re-type the book path each
# time. Stored outside any experiment dir so it survives across books.
LAST_RUN_FILE = Path.home() / ".autofilm" / "last_run.json"


def _load_last_run() -> dict[str, Any] | None:
    if not LAST_RUN_FILE.exists():
        return None
    try:
        return json.loads(LAST_RUN_FILE.read_text())
    except Exception:
        return None


def _save_last_run(config: dict[str, Any]) -> None:
    """Persist a slim subset of the start config so we don't leak large
    file lists or transient runtime fields."""
    slim = {
        "book_path":        config.get("book_path"),
        "book_filename":    config.get("book_filename"),
        "book_slug":        config.get("book_slug"),
        "iterations":       config.get("iterations"),
        "target_loss":      config.get("target_loss"),
        "threshold":        config.get("threshold"),
        "director":         config.get("director"),
        "cinematographer":  config.get("cinematographer"),
        "saved_at":         time.time(),
    }
    try:
        LAST_RUN_FILE.parent.mkdir(parents=True, exist_ok=True)
        LAST_RUN_FILE.write_text(json.dumps(slim, indent=2))
    except Exception:
        pass


# ── Stage definitions (must match produce.py / run_loop.py output) ───────────

PIPELINE_STAGES = [
    {"id": "1",     "label": "parse script"},
    {"id": "1.5",   "label": "format screenplay"},
    {"id": "2",     "label": "cast + locations"},
    {"id": "3",     "label": "lookbook"},
    {"id": "4",     "label": "reference images"},
    {"id": "5",     "label": "shot list"},
    {"id": "6",     "label": "music"},
    {"id": "6.5",   "label": "narration"},
    {"id": "7",     "label": "first frames"},
    {"id": "8",     "label": "video"},
    {"id": "9",     "label": "edit decisions"},
    {"id": "10",    "label": "compile + grade"},
    {"id": "eval",  "label": "evaluate"},
    {"id": "bible", "label": "bible"},
]


_STAGE_RE     = re.compile(r"\bStage\s+(\d+(?:\.\d+)?):\s*(.+)")
_ITER_RE      = re.compile(r"ITERATION\s+(\d+)/(\d+)\s+exp\s*=\s*(\S+)")
_EVAL_RE      = re.compile(r"\bevaluating film\b|\bevaluate:\s+\d+s\b", re.IGNORECASE)
_BIBLE_RE     = re.compile(r"production_bible\.json|Production bible PDF", re.IGNORECASE)
_FILM_LOSS_RE = re.compile(r"film_loss\s*=\s*([0-9.]+)")
_DONE_RE      = re.compile(r"Loop completed:\s*(\d+)/(\d+)|Target film_loss")


def parse_progress(log_lines: list[str]) -> dict[str, Any]:
    """Walk the log buffer and extract loop-level + stage-level progress.

    Returns the latest iteration index, latest stage, set of stages
    completed in the current iteration, and most-recent film_loss.
    Designed to be cheap (one pass).
    """
    cur_iter:   int | None = None
    total_iter: int | None = None
    cur_exp:    str | None = None
    cur_stage:  str | None = None
    completed_stages: list[str] = []
    last_film_loss: float | None = None
    finished = False

    for line in log_lines:
        m = _ITER_RE.search(line)
        if m:
            cur_iter   = int(m.group(1))
            total_iter = int(m.group(2))
            cur_exp    = m.group(3)
            cur_stage  = None
            completed_stages = []
            continue

        m = _STAGE_RE.search(line)
        if m:
            stage_id = m.group(1)
            if cur_stage and cur_stage not in completed_stages:
                completed_stages.append(cur_stage)
            cur_stage = stage_id
            continue

        if _EVAL_RE.search(line):
            if cur_stage and cur_stage not in completed_stages:
                completed_stages.append(cur_stage)
            cur_stage = "eval"
            continue

        if _BIBLE_RE.search(line):
            if cur_stage and cur_stage != "bible" and cur_stage not in completed_stages:
                completed_stages.append(cur_stage)
            cur_stage = "bible"
            continue

        m = _FILM_LOSS_RE.search(line)
        if m:
            try:
                last_film_loss = float(m.group(1))
            except ValueError:
                pass

        if _DONE_RE.search(line):
            finished = True

    return {
        "current_iteration":  cur_iter,
        "total_iterations":   total_iter,
        "current_exp":        cur_exp,
        "current_stage":      cur_stage,
        "completed_stages":   completed_stages,
        "last_film_loss":     last_film_loss,
        "finished":           finished,
    }


# ── Run state ────────────────────────────────────────────────────────────────

class RunState:
    """In-memory state for a single subprocess (run-loop or smoke test)."""
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

    def snapshot(self, include_progress: bool = True) -> dict[str, Any]:
        with self.lock:
            buf = self.log_buffer.copy()
        progress = parse_progress(buf) if (include_progress and buf) else {}
        return {
            "status":       self.status,
            "config":       self.config,
            "started_at":   self.started_at,
            "finished_at":  self.finished_at,
            "exit_code":    self.exit_code,
            "is_running":   self.process is not None and self.process.poll() is None,
            "log_tail":     buf[-600:],
            "log_total":    len(buf),
            "progress":     progress,
        }


run_state   = RunState()
smoke_state = RunState()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_book_slug(book_path: Path) -> str:
    """Match prepare._book_slug() so user moodboards land where the
    pipeline reads them."""
    return "".join(
        c.lower() if c.isalnum() else "_" for c in book_path.stem
    ).strip("_") or "book"


def _spawn_subprocess(
    state: RunState,
    cmd: list[str],
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> None:
    """Start a subprocess in a background thread; tee output into log_buffer."""
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
    """Resolve ``candidate`` relative to ``root``; raise 404 if it escapes."""
    candidate = candidate.lstrip("/")
    full = (root / candidate).resolve()
    if not full.is_relative_to(root.resolve()):
        abort(404)
    if not full.exists():
        abort(404)
    return full


def _load_run_config(exp_path: Path) -> dict[str, Any] | None:
    f = exp_path / "run_config.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text())
    except Exception:
        return None


# ── Flask app ─────────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder=None)


@app.route("/")
def index() -> Any:
    return send_from_directory(UI_DIR, "index.html")


@app.route("/static/<path:fname>")
def static_files(fname: str) -> Any:
    return send_from_directory(UI_DIR / "static", fname)


# ── State ────────────────────────────────────────────────────────────────────

@app.get("/api/state")
def api_state() -> Any:
    """Aggregate state: run + smoke + lightweight experiment list.

    Full per-experiment data is NOT inlined (this endpoint is polled
    every few seconds). Use /api/experiment/<book>/<exp> for full data.
    """
    items: list[dict[str, Any]] = []
    for exp_path in sorted(iter_all_experiments(), key=lambda p: p.stat().st_mtime, reverse=True):
        items.append({
            "exp_id":      exp_path.name,
            "book_slug":   exp_path.parent.name,
            "qualified":   f"{exp_path.parent.name}/{exp_path.name}",
            "mtime":       exp_path.stat().st_mtime,
            "has_final":   (exp_path / "final.mp4").exists(),
            "has_metric":  (exp_path / "metric.json").exists(),
        })

    return jsonify({
        "run":              run_state.snapshot(),
        "smoke":            smoke_state.snapshot(),
        "experiments":      items[:60],
        "stages":           PIPELINE_STAGES,
        "last_run_config":  _load_last_run(),
    })


@app.get("/api/experiment/<book_slug>/<exp_id>")
def api_experiment(book_slug: str, exp_id: str) -> Any:
    """Full data for one experiment.

    The bible is built ON THE FLY from the filesystem (not read from the
    on-disk production_bible.json) so the UI sees content the moment it
    appears under the experiment dir — no need to wait for the pipeline
    to write the manifest at the end.

    Cost is estimated from prompts.json via cost.aggregate_costs().
    """
    from prepare import Experiment  # late import to avoid circular
    from production_bible import build_production_bible_dict
    from cost import aggregate_costs

    exp_root = EXPERIMENTS_DIR / book_slug / exp_id
    if not exp_root.is_dir():
        abort(404)

    out: dict[str, Any] = {
        "qualified":   f"{book_slug}/{exp_id}",
        "exp_id":      exp_id,
        "book_slug":   book_slug,
        "has_final":   (exp_root / "final.mp4").exists(),
    }

    # Live bible (filesystem-scanned, always current).
    try:
        exp = Experiment(exp_id=exp_id, root=exp_root)
        out["bible"] = build_production_bible_dict(exp)
    except Exception as e:  # noqa: BLE001
        out["bible_error"] = str(e)

    # Run config (defaults for the form when this exp is selected).
    rc = _load_run_config(exp_root)
    if rc:
        out["run_config"] = rc

    # Cost breakdown (per-model + per-stage, plus cumulative across the
    # iteration chain if this exp has parents).
    try:
        out["cost"] = aggregate_costs(exp_root)
    except Exception as e:  # noqa: BLE001
        out["cost_error"] = str(e)

    return jsonify(out)


@app.get("/api/log/run")
def api_log_run() -> Any:
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
      book_path        — server-local path to the book PDF
      book_file        — uploaded PDF (alternative to book_path)
      iterations       — int (default 3)
      target_loss      — float (default 0.15)
      threshold        — 'low' | 'medium' | 'high' (default 'medium')
      director         — string, optional
      cinematographer  — string, optional
      moodboards       — list of file uploads, optional

    At least one of book_path / book_file is required.
    """
    if run_state.snapshot(include_progress=False)["is_running"]:
        return jsonify({"error": "A run is already in progress."}), 409

    # ── Resolve book_path: either a server-local path or an upload ──
    book_path: Path | None = None

    book_file = request.files.get("book_file")
    if book_file and book_file.filename:
        name = Path(book_file.filename).name
        if Path(name).suffix.lower() != ".pdf":
            return jsonify({"error": "book_file must be a .pdf"}), 400
        dest = UPLOAD_DIR / name
        # Dedupe with numeric suffix if a different-sized file already
        # exists at this name; never silently overwrite.
        i = 1
        while dest.exists() and dest.stat().st_size != (book_file.content_length or 0):
            dest = UPLOAD_DIR / f"{Path(name).stem}_{i}.pdf"
            i += 1
        book_file.save(dest)
        book_path = dest
    else:
        book_path_str = (request.form.get("book_path") or "").strip()
        if book_path_str:
            book_path = Path(book_path_str).expanduser()

    if book_path is None:
        return jsonify({"error": "book_path or book_file is required"}), 400
    if not book_path.is_file():
        return jsonify({"error": f"book file not found: {book_path}"}), 400

    iterations  = max(1, min(int(request.form.get("iterations", "3")), 20))
    target_loss = float(request.form.get("target_loss", "0.15"))
    threshold   = request.form.get("threshold", "medium")
    if threshold not in ("low", "medium", "high"):
        threshold = "medium"

    director         = (request.form.get("director") or "").strip()
    cinematographer  = (request.form.get("cinematographer") or "").strip()

    # ── Save uploaded moodboards per-book ──
    book_slug   = _safe_book_slug(book_path)
    user_mb_dir = EXPERIMENTS_DIR / book_slug / "user_moodboards"
    user_mb_dir.mkdir(parents=True, exist_ok=True)
    saved_moodboards: list[str] = []
    for f in request.files.getlist("moodboards"):
        if not f or not f.filename:
            continue
        name = Path(f.filename).name
        if Path(name).suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            continue
        dest = user_mb_dir / name
        f.save(dest)
        saved_moodboards.append(str(dest.relative_to(EXPERIMENTS_DIR)))

    # ── Persist the run config so the dropdown can re-load it later ──
    config = {
        "book_path":         str(book_path),
        "book_filename":     book_path.name,
        "book_slug":         book_slug,
        "iterations":        iterations,
        "target_loss":       target_loss,
        "threshold":         threshold,
        "director":          director,
        "cinematographer":   cinematographer,
        "moodboards_saved":  saved_moodboards,
        "started_at":        time.time(),
    }
    # The new exp dir doesn't exist yet; write the config to a "next-run"
    # cache that produce.py picks up and copies into the exp dir on creation.
    pending_path = EXPERIMENTS_DIR / book_slug / "_pending_run_config.json"
    pending_path.parent.mkdir(parents=True, exist_ok=True)
    pending_path.write_text(json.dumps(config, indent=2))

    cmd = [
        sys.executable, "run_loop.py",
        "--iterations", str(iterations),
        "--target",     f"{target_loss}",
        "--threshold",  threshold,
    ]
    env: dict[str, str] = {
        "BOOK_PDF_PATH":              str(book_path),
        "AUTOFILM_PENDING_RUN_CONFIG": str(pending_path),
    }
    if director:
        env["DIRECTOR"] = director
    if cinematographer:
        env["CINEMATOGRAPHER"] = cinematographer

    run_state.config = config
    _spawn_subprocess(run_state, cmd, env=env)

    # Persist these settings as the new defaults for the next page load.
    _save_last_run(config)

    return jsonify({"ok": True, "config": config})


@app.post("/api/stop")
def api_stop() -> Any:
    with run_state.lock:
        if run_state.process and run_state.process.poll() is None:
            try:
                run_state.process.send_signal(signal.SIGTERM)
            except Exception as e:  # noqa: BLE001
                return jsonify({"error": str(e)}), 500
            return jsonify({"ok": True, "stopped": True})
    return jsonify({"ok": True, "stopped": False, "note": "no run in progress"})


# ── API smoke test ───────────────────────────────────────────────────────────

@app.post("/api/smoke-test")
def api_smoke_test() -> Any:
    if smoke_state.snapshot(include_progress=False)["is_running"]:
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


# ── Artifact serving ─────────────────────────────────────────────────────────

@app.get("/api/artifact/<book_slug>/<exp_id>/<path:relpath>")
def api_artifact(book_slug: str, exp_id: str, relpath: str) -> Any:
    exp_root = EXPERIMENTS_DIR / book_slug / exp_id
    if not exp_root.is_dir():
        abort(404)
    full = _safe_relative_path(exp_root, relpath)
    return send_file(full)


@app.get("/api/health")
def health() -> Any:
    return jsonify({"ok": True, "experiments_dir": str(EXPERIMENTS_DIR)})


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    host = os.environ.get("AUTOFILM_UI_HOST", "127.0.0.1")
    port = int(os.environ.get("AUTOFILM_UI_PORT", "5174"))
    print(f"autofilm UI on http://{host}:{port}")
    print(f"  experiments dir: {EXPERIMENTS_DIR}")
    print(f"  upload dir:      {UPLOAD_DIR}")
    app.run(host=host, port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
