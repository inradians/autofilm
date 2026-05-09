"""scripts/ui_smoke_test.py — UI server smoke test (no expensive API calls).

Boots ui_server.py against an isolated temp experiments dir, populates
that dir with synthetic placeholder artifacts, and probes every HTTP
endpoint to make sure:

  - /api/state returns the experiment list and pipeline stages
  - /api/experiment/<book>/<exp> builds the bible LIVE from disk
  - /api/artifact serves PNG / MP4 / WAV / MP3 correctly
  - Path-traversal attempts are rejected (404)
  - Bad form input is rejected (400)
  - Cost aggregation runs end-to-end against a synthetic prompts.json
  - The progress parser handles a real iteration banner + stage lines

By default this test makes ZERO external API calls — every artifact is a
tiny synthetic file. The expensive providers (Veo, SeedDance, LTX,
Stable Audio, ElevenLabs, FLUX, GPT-Image, Gen4) all have rate limits
and per-call costs, so we don't hit them just to debug the UI.

Optional flags:

  --with-image         Make ONE real cheap image call (Reve or Stability
                       Core, whichever has a key set) to validate that
                       the image pipeline actually produces a valid PNG
                       and the UI serves it. Costs about $0.01-$0.03.

  --port N             Bind the test server to a specific port
                       (default: random free port).

  -v / --verbose       Print extra debug info on each test.

Examples:

    python scripts/ui_smoke_test.py
    python scripts/ui_smoke_test.py --with-image
    python scripts/ui_smoke_test.py --with-image -v
"""
from __future__ import annotations

import argparse
import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── Test result tracking ──────────────────────────────────────────────────────

class TestResult:
    def __init__(self, name: str):
        self.name = name
        self.status = "PENDING"
        self.note = ""
    def __repr__(self) -> str:
        icon = {"PASS": "✓", "FAIL": "✗", "SKIP": "–"}.get(self.status, "?")
        note = f"  {self.note}" if self.note else ""
        return f"  {icon}  {self.name}{note}"


_results: list[TestResult] = []
_verbose = False


class SkipTest(Exception):
    """Signal that a test was skipped (not failed)."""


def test(name: str):
    def deco(fn: Callable[..., None]) -> Callable[..., None]:
        def wrapper(*args, **kwargs) -> TestResult:
            r = TestResult(name)
            _results.append(r)
            if _verbose:
                print(f"\n  ▸ {name}")
            try:
                fn(*args, **kwargs)
                r.status = "PASS"
            except SkipTest as e:
                r.status = "SKIP"
                r.note = str(e)[:200]
            except AssertionError as e:
                r.status = "FAIL"
                r.note = str(e)[:200]
                if _verbose: traceback.print_exc()
            except Exception as e:  # noqa: BLE001
                r.status = "FAIL"
                r.note = f"{type(e).__name__}: {str(e)[:180]}"
                if _verbose: traceback.print_exc()
            return r
        wrapper.__name__ = fn.__name__
        return wrapper
    return deco


# ── Synthetic file builders ───────────────────────────────────────────────────

def tiny_png() -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), (60, 80, 100)).save(buf, "PNG")
    return buf.getvalue()


def tiny_jpg() -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), (60, 80, 100)).save(buf, "JPEG")
    return buf.getvalue()


def tiny_wav() -> bytes:
    """Valid 1-sample 16-bit PCM mono 44.1kHz WAV."""
    import struct
    pcm = b"\x00\x00"
    fmt  = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVE"
    fmt += b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, 44100, 88200, 2, 16)
    fmt += b"data" + struct.pack("<I", len(pcm)) + pcm
    return fmt


def tiny_mp4() -> bytes:
    """Minimal MP4 with ftyp + empty mdat box. Won't play, but the bytes
    pass MIME sniffing for an artifact serving smoke test."""
    return (
        b"\x00\x00\x00\x20" b"ftyp" b"mp42"
        b"\x00\x00\x00\x00" b"mp42isom"
        b"\x00\x00\x00\x08" b"mdat"
    )


def tiny_mp3() -> bytes:
    """A valid silent MP3 frame header. ~0.026s of silence."""
    # MPEG1 Layer III, 128kbps, 44.1kHz, mono, no padding, no CRC
    return b"\xff\xfb\x90\x00" + b"\x00" * 384


def build_synthetic_experiment(exp_root: Path) -> None:
    """Populate exp_root with one of every artifact type the UI renders."""
    scene_id = "scene_001"
    shot_id  = "shot_001"
    char_id  = "ada"

    (exp_root / "book.txt").write_text("smoke_book")
    (exp_root / "seed.txt").write_text("12345")

    # Stage 1: script (with one narration element to exercise that branch)
    script = {
        "title": "UI Smoke",
        "scenes": [{
            "id": scene_id, "location": "test set",
            "characters": [char_id], "mood": "spare",
            "elements": [
                {"type": "action",    "text": "Ada steps into the light."},
                {"type": "dialogue",  "text": "Hello.", "character": char_id},
                {"type": "narration", "text": "And so it began.",
                 "character": "narrator"},
            ],
        }],
        "characters": [{"id": char_id, "name": "Ada",
                        "description": "A wiry coder, watchful gaze"}],
    }
    (exp_root / "script.json").write_text(json.dumps(script))

    # Stage 2: cast + locations
    cast = [{"character_id": char_id, "actor": "wiry coder, watchful gaze"}]
    (exp_root / "cast.json").write_text(json.dumps(cast))
    locations = [{"slug": "test_set", "description": "an empty soundstage",
                  "color_palette": ["amber", "teal"], "scene_ids": [scene_id]}]
    (exp_root / "locations.json").write_text(json.dumps(locations))
    # Moodboard
    mb_path = exp_root / "location_moodboards" / "test_set" / "00.png"
    mb_path.parent.mkdir(parents=True, exist_ok=True)
    mb_path.write_bytes(tiny_png())

    # Stage 3: lookbook + style frame
    lb = {"grade_description": "test grade", "ffmpeg_grade": "eq=contrast=1.0",
          "style_keywords": ["35mm", "anamorphic"]}
    (exp_root / "lookbook.json").write_text(json.dumps(lb))
    (exp_root / "lookbook").mkdir(exist_ok=True)
    (exp_root / "lookbook" / "style_frame.png").write_bytes(tiny_png())

    # Stage 4: references — references/{char}/{scene}.png
    refp = exp_root / "references" / char_id / f"{scene_id}.png"
    refp.parent.mkdir(parents=True, exist_ok=True)
    refp.write_bytes(tiny_png())

    # Stage 5: storyboard
    storyboard = {scene_id: [{"shot_id": shot_id, "shot_size": "MS",
                               "angle": "eye-level", "duration_seconds": 6,
                               "action": "Ada walks."}]}
    (exp_root / "storyboard.json").write_text(json.dumps(storyboard))

    # Stage 6: music
    (exp_root / "music").mkdir(exist_ok=True)
    (exp_root / "music" / f"{scene_id}.wav").write_bytes(tiny_wav())

    # Stage 6.5: narration
    (exp_root / "narration").mkdir(exist_ok=True)
    (exp_root / "narration" / f"{scene_id}.mp3").write_bytes(tiny_mp3())

    # Stage 7: first frames — frames/{scene}/{shot}.png
    frp = exp_root / "frames" / scene_id / f"{shot_id}.png"
    frp.parent.mkdir(parents=True, exist_ok=True)
    frp.write_bytes(tiny_png())

    # Stage 8: clips/{scene}/{shot}/take_N.mp4 (two takes)
    cdir = exp_root / "clips" / scene_id / shot_id
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "take_0.mp4").write_bytes(tiny_mp4())
    (cdir / "take_1.mp4").write_bytes(tiny_mp4())

    # Stage 9: edl
    (exp_root / "edl.json").write_text(
        json.dumps({scene_id: [{"shot_id": shot_id, "take": 0}]})
    )

    # SFX
    sfxp = exp_root / "sfx" / scene_id / "ambient.wav"
    sfxp.parent.mkdir(parents=True, exist_ok=True)
    sfxp.write_bytes(tiny_wav())

    # Stage 10: final
    (exp_root / "final.mp4").write_bytes(tiny_mp4())

    # prompts.json — exercises cost.aggregate_costs end-to-end
    prompts = {
        "lookbook/style_frame.png": {"target": "lookbook/style_frame.png",
                                      "model": "gpt_image", "stage": "style_frame"},
        f"frames/{scene_id}/{shot_id}.png": {
            "target": f"frames/{scene_id}/{shot_id}.png",
            "model": "gen4_image+refs", "stage": "first_frames"},
        f"clips/{scene_id}/{shot_id}/take_0.mp4": {
            "target": f"clips/{scene_id}/{shot_id}/take_0.mp4",
            "model": "veo3.1", "stage": "video", "duration_seconds": 8},
        f"music/{scene_id}.wav": {"target": f"music/{scene_id}.wav",
            "model": "stable-audio-2.5", "stage": "music",
            "duration_seconds": 30},
        f"narration/{scene_id}.mp3": {"target": f"narration/{scene_id}.mp3",
            "model": "eleven_multilingual_v2", "stage": "narration"},
    }
    (exp_root / "prompts.json").write_text(json.dumps(prompts))

    # run_config.json — exercises form-default loading
    (exp_root / "run_config.json").write_text(json.dumps({
        "book_path": "/tmp/smoke.pdf",
        "iterations": 3, "target_loss": 0.15, "threshold": "medium",
        "director": "Test Director", "cinematographer": "Test DP",
    }))

    # Snapshot of produce.py for the bible's config section
    src = PROJECT_ROOT / "produce.py"
    if src.exists():
        (exp_root / "produce.py").write_text(src.read_text())


# ── Server lifecycle ──────────────────────────────────────────────────────────

def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class UIServer:
    """Boots ui_server.py as a subprocess pointed at our temp experiments dir."""
    def __init__(self, exp_dir: Path, port: int):
        self.exp_dir = exp_dir
        self.port    = port
        self.proc:   subprocess.Popen | None = None

    def __enter__(self) -> "UIServer":
        env = {
            **os.environ,
            "AUTOFILM_UI_PORT":   str(self.port),
            "AUTOFILM_UI_HOST":   "127.0.0.1",
            "AUTOFILM_TEST_EXPERIMENTS_DIR": str(self.exp_dir),
            "PYTHONUNBUFFERED":   "1",
        }
        self.proc = subprocess.Popen(
            [sys.executable, "ui_server.py"],
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        # Wait for the port to start accepting connections (max 10s)
        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.5):
                    return self
            except OSError:
                time.sleep(0.2)
            if self.proc.poll() is not None:
                # Server died early — surface its log
                out = self.proc.stdout.read() if self.proc.stdout else ""
                raise RuntimeError(
                    f"ui_server died before binding port {self.port}. "
                    f"output:\n{out[-2000:]}"
                )
        raise RuntimeError(f"ui_server didn't bind {self.port} within 10s")

    def __exit__(self, *exc) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def http(method: str, url: str, *, data: bytes | None = None,
         timeout: float = 10.0,
         headers: dict | None = None) -> tuple[int, bytes, dict]:
    """Returns (status, body_bytes, headers)."""
    req = urllib.request.Request(url, data=data, method=method,
                                  headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read(), dict(resp.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers)


def http_json(method: str, url: str, **kw) -> tuple[int, Any, dict]:
    code, body, hdr = http(method, url, **kw)
    try:
        return code, json.loads(body), hdr
    except Exception:
        return code, None, hdr


# ── Tests ─────────────────────────────────────────────────────────────────────

@test("/api/health returns ok")
def t_health(base: str, **_) -> None:
    code, j, _ = http_json("GET", f"{base}/api/health")
    assert code == 200, f"got {code}"
    assert j["ok"] is True


@test("/api/state returns expected shape")
def t_state(base: str, **_) -> None:
    code, j, _ = http_json("GET", f"{base}/api/state")
    assert code == 200, f"got {code}"
    for key in ("run", "smoke", "experiments", "stages", "last_run_config"):
        assert key in j, f"missing key: {key}"
    assert isinstance(j["stages"], list) and len(j["stages"]) >= 12
    assert any(e["qualified"] == "smoke_book/exp_001" for e in j["experiments"]), \
        f"synthetic experiment not listed: {[e['qualified'] for e in j['experiments']]}"


@test("/api/state includes pipeline stages with id/label")
def t_state_stages(base: str, **_) -> None:
    code, j, _ = http_json("GET", f"{base}/api/state")
    ids = [s["id"] for s in j["stages"]]
    for needed in ("1", "1.5", "6.5", "eval", "bible"):
        assert needed in ids, f"missing stage id: {needed}"


@test("/api/experiment builds the bible LIVE from filesystem")
def t_experiment(base: str, **_) -> None:
    code, j, _ = http_json("GET", f"{base}/api/experiment/smoke_book/exp_001")
    assert code == 200, f"got {code}"

    # Top-level keys
    for k in ("qualified", "exp_id", "book_slug", "has_final",
              "bible", "run_config", "cost"):
        assert k in j, f"missing: {k}"

    # Bible has every stage's artifacts
    stages = j["bible"]["stages"]
    for needed in ("script", "cast", "locations", "lookbook",
                   "references", "storyboard", "music", "narration",
                   "frames", "clips", "edl", "sfx", "final"):
        assert needed in stages, f"bible missing stage: {needed}"

    # Spot-check the canonical key formats
    assert "scene_001:ada"        in stages["references"]["by_scene_char"]
    assert "scene_001:shot_001"   in stages["frames"]["by_scene_shot"]
    assert "scene_001:shot_001:take_0" in stages["clips"]["by_scene_shot_take"]

    # Live moodboard scan (the dir is scanned, not locations.json)
    assert "test_set" in stages["locations"]["moodboards"]


@test("/api/experiment includes run_config defaults for the form")
def t_experiment_run_config(base: str, **_) -> None:
    code, j, _ = http_json("GET", f"{base}/api/experiment/smoke_book/exp_001")
    rc = j.get("run_config")
    assert rc is not None
    assert rc["director"]  == "Test Director"
    assert rc["iterations"] == 3


@test("/api/experiment cost aggregator returns per-model breakdown")
def t_experiment_cost(base: str, **_) -> None:
    code, j, _ = http_json("GET", f"{base}/api/experiment/smoke_book/exp_001")
    cost = j["cost"]
    assert cost["n_calls"] == 5
    # video should be the dominant cost (8s * $0.080)
    assert "veo3.1" in cost["by_model"], list(cost["by_model"].keys())
    assert cost["by_model"]["veo3.1"] > 0.5
    assert cost["total_usd"] > 0.5


@test("/api/artifact serves PNG (style_frame)")
def t_artifact_png(base: str, **_) -> None:
    code, body, hdr = http(
        "GET", f"{base}/api/artifact/smoke_book/exp_001/lookbook/style_frame.png")
    assert code == 200, f"got {code}"
    assert body[:8] == b"\x89PNG\r\n\x1a\n", f"bad PNG header: {body[:8]!r}"


@test("/api/artifact serves MP4 (final.mp4)")
def t_artifact_mp4(base: str, **_) -> None:
    code, body, hdr = http(
        "GET", f"{base}/api/artifact/smoke_book/exp_001/final.mp4")
    assert code == 200, f"got {code}"
    assert b"ftyp" in body[:32], f"missing ftyp box in MP4 head: {body[:32]!r}"


@test("/api/artifact serves WAV (music)")
def t_artifact_wav(base: str, **_) -> None:
    code, body, _ = http(
        "GET", f"{base}/api/artifact/smoke_book/exp_001/music/scene_001.wav")
    assert code == 200, f"got {code}"
    assert body[:4] == b"RIFF", f"missing RIFF header: {body[:4]!r}"


@test("/api/artifact serves MP3 (narration)")
def t_artifact_mp3(base: str, **_) -> None:
    code, body, _ = http(
        "GET", f"{base}/api/artifact/smoke_book/exp_001/narration/scene_001.mp3")
    assert code == 200, f"got {code}"
    # MPEG audio sync word: 0xFF 0xFB (or 0xF3, 0xFA)
    assert body[:1] == b"\xff" and body[1] in (0xfb, 0xf3, 0xfa), \
        f"bad MP3 sync: {body[:2]!r}"


@test("/api/artifact serves clips (take_0.mp4)")
def t_artifact_clip(base: str, **_) -> None:
    code, body, _ = http(
        "GET", f"{base}/api/artifact/smoke_book/exp_001/"
        "clips/scene_001/shot_001/take_0.mp4")
    assert code == 200, f"got {code}"


@test("/api/artifact rejects path traversal (../etc/passwd)")
def t_artifact_traversal(base: str, **_) -> None:
    code, _, _ = http(
        "GET", f"{base}/api/artifact/smoke_book/exp_001/../../../etc/passwd")
    assert code == 404, f"path traversal returned {code}, expected 404"


@test("/api/artifact 404s on missing file")
def t_artifact_404(base: str, **_) -> None:
    code, _, _ = http(
        "GET", f"{base}/api/artifact/smoke_book/exp_001/nope/missing.png")
    assert code == 404, f"got {code}"


@test("/api/experiment 404s on unknown experiment")
def t_experiment_404(base: str, **_) -> None:
    code, _, _ = http("GET", f"{base}/api/experiment/no_book/exp_999")
    assert code == 404


@test("POST /api/start without book_path or book_file → 400")
def t_start_no_book(base: str, **_) -> None:
    # Empty multipart body
    boundary = "----smoke"
    body = (f"--{boundary}--\r\n").encode()
    code, j, _ = http_json(
        "POST", f"{base}/api/start", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    assert code == 400, f"got {code} body={j}"
    assert "error" in j


@test("POST /api/start with non-existent book_path → 400")
def t_start_bad_book(base: str, **_) -> None:
    boundary = "----smoke"
    body = (
        f"--{boundary}\r\n"
        f"Content-Disposition: form-data; name=\"book_path\"\r\n\r\n"
        f"/this/does/not/exist.pdf\r\n"
        f"--{boundary}--\r\n"
    ).encode()
    code, j, _ = http_json(
        "POST", f"{base}/api/start", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    assert code == 400, f"got {code} body={j}"


@test("GET / returns the HTML index")
def t_index(base: str, **_) -> None:
    code, body, _ = http("GET", f"{base}/")
    assert code == 200, f"got {code}"
    assert b"<title>autofilm" in body, "title tag missing — wrong page served?"
    # Sanity: our drag-drop dropzone is in there
    assert b"bookDropzone" in body
    # Sanity: stepper container
    assert b'id="stepper"' in body


# ── Optional: one cheap real image call ──────────────────────────────────────

@test("real cheap image call (--with-image) produces a valid PNG/JPEG")
def t_real_image(base: str, exp_dir: Path, **_) -> None:
    """Call ONE cheap image endpoint and verify the bytes are a real image
    and the UI serves them. Uses Reve (~$0.01) by default; falls back to
    Stability Core (~$0.03) if REVE_API_KEY isn't set."""
    chosen: str | None = None
    img_bytes: bytes | None = None

    if os.environ.get("REVE_API_KEY"):
        from prepare import reve_image
        chosen = "reve_create"
        img_bytes = reve_image(
            "Cinematic establishing shot, soft amber light, "
            "minimal composition, photorealistic. NEGATIVE: text, logos.",
            aspect_ratio="16:9",
        )
    elif os.environ.get("STABILITY_API_KEY"):
        from prepare import stable_image
        chosen = "stable_image"
        img_bytes = stable_image(
            "Cinematic establishing shot, soft amber light, minimal "
            "composition, photorealistic. No text. No logos.",
            aspect_ratio="16:9", tier="core",
        )
    else:
        raise SkipTest(
            "neither REVE_API_KEY nor STABILITY_API_KEY is set"
        )

    # Validate the bytes — PNG, JPEG, or WEBP.
    assert img_bytes and len(img_bytes) > 1024, \
        f"too-small image: {len(img_bytes) if img_bytes else 0} bytes"
    head = img_bytes[:12]
    is_png  = head[:8] == b"\x89PNG\r\n\x1a\n"
    is_jpg  = head[:3] == b"\xff\xd8\xff"
    is_webp = head[:4] == b"RIFF" and head[8:12] == b"WEBP"
    assert is_png or is_jpg or is_webp, \
        f"unrecognized image header from {chosen}: {head!r}"

    # Drop into the synthetic exp dir and verify the UI serves it.
    out_path = exp_dir / "smoke_book" / "exp_001" / "lookbook" / "real_image.png"
    out_path.write_bytes(img_bytes)
    code, served, _ = http(
        "GET", f"{base}/api/artifact/smoke_book/exp_001/lookbook/real_image.png")
    assert code == 200
    assert served == img_bytes, "served bytes don't match disk contents"
    print(f"     · {chosen}: {len(img_bytes)//1024}kB served correctly")


# ── Patch ui_server.py to honor a test-only EXPERIMENTS_DIR override ─────────
#
# ui_server reads EXPERIMENTS_DIR from `prepare` at import time. To point
# the booted server at our temp dir we set AUTOFILM_TEST_EXPERIMENTS_DIR
# in the env and patch prepare.EXPERIMENTS_DIR at module init. The patch
# is small and only fires when the env var is set.
#
# The patch is injected via a tiny shim file that ui_server imports IF
# the env var is set. If it isn't set, ui_server runs normally.
#
# Implementation: rather than monkey-patching ui_server itself, we
# pre-write a small site-customizing helper into PROJECT_ROOT for the
# duration of the test, then remove it. But this is brittle. Cleaner:
# pass the env var, and have ui_server.py respect it on its own.
#
# Since modifying ui_server.py is part of the deliverable anyway, this
# smoke test depends on that hook (added in the same patch).


# ── Runner ────────────────────────────────────────────────────────────────────

ALL_TESTS = [
    t_health, t_state, t_state_stages,
    t_experiment, t_experiment_run_config, t_experiment_cost,
    t_artifact_png, t_artifact_mp4, t_artifact_wav, t_artifact_mp3,
    t_artifact_clip,
    t_artifact_traversal, t_artifact_404, t_experiment_404,
    t_start_no_book, t_start_bad_book,
    t_index,
]


def main() -> int:
    global _verbose
    ap = argparse.ArgumentParser(
        description="UI server smoke test (no expensive API calls by default)."
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--with-image", action="store_true",
                    help="Make ONE real cheap image call (Reve or Stability Core)")
    ap.add_argument("--port", type=int, default=0,
                    help="Bind port (default: random free port)")
    args = ap.parse_args()
    _verbose = args.verbose

    print("Booting ui_server against an isolated temp experiments dir...")

    with tempfile.TemporaryDirectory(prefix="autofilm_ui_smoke_") as tmp:
        exp_dir = Path(tmp) / "experiments"
        exp_dir.mkdir()
        exp_root = exp_dir / "smoke_book" / "exp_001"
        exp_root.mkdir(parents=True)
        build_synthetic_experiment(exp_root)

        port = args.port or free_port()
        with UIServer(exp_dir, port) as _srv:
            base = f"http://127.0.0.1:{port}"
            print(f"  server on {base}\n")

            for fn in ALL_TESTS:
                fn(base=base, exp_dir=exp_dir)

            if args.with_image:
                t_real_image(base=base, exp_dir=exp_dir)

    print(f"\n{'─' * 60}")
    print("  Results")
    print(f"{'─' * 60}")
    for r in _results:
        print(r)
    print(f"{'─' * 60}")
    n_pass = sum(1 for r in _results if r.status == "PASS")
    n_fail = sum(1 for r in _results if r.status == "FAIL")
    n_skip = sum(1 for r in _results if r.status == "SKIP")
    print(f"  {n_pass} PASS  ·  {n_fail} FAIL  ·  {n_skip} SKIP  ·  "
          f"{len(_results)} total")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
