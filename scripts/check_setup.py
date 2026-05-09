"""scripts/check_setup.py — verify autofilm is configured and ready to run.

Pings each provider with the cheapest possible call (or a free metadata
endpoint where one exists), checks ffmpeg, checks the book PDF.

Total cost of a clean run: well under a cent. Total time: ~10 seconds.

Usage:
    python scripts/check_setup.py

Exit code 0 if everything is OK, 1 if any check failed.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ---- pretty printing ----------------------------------------------------
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
DIM = "\033[2m"
RESET = "\033[0m"


def line(label: str, status: str, detail: str) -> None:
    color = {
        "OK":   GREEN,
        "WARN": YELLOW,
        "SKIP": DIM,
    }.get(status, RED)
    print(f"  {label:<22s} {color}{status:<5s}{RESET}{DIM} {detail}{RESET}")


# ---- per-provider checks -----------------------------------------------
def check_anthropic() -> bool:
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key or key.startswith("sk-ant-...") or key == "":
        line("ANTHROPIC_API_KEY", "FAIL", "missing — see SETUP.md §4a")
        return False
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        # 1-token reachability check, ~$0.00003.
        client.messages.create(
            model="claude-opus-4-7",
            max_tokens=1,
            messages=[{"role": "user", "content": "."}],
        )
        line("ANTHROPIC_API_KEY", "OK", "claude-opus-4-7 reachable")
        return True
    except Exception as e:  # noqa: BLE001
        msg = str(e).split("\n")[0][:120]
        line("ANTHROPIC_API_KEY", "FAIL", msg)
        return False


def check_runway() -> bool:
    key = os.getenv("RUNWAYML_API_SECRET", "")
    if not key or key.startswith("key_..."):
        line("RUNWAYML_API_SECRET", "FAIL", "missing — see SETUP.md §4b")
        return False
    try:
        import httpx
        # Runway exposes an /v1/organization endpoint that returns the
        # caller's org info + credit balance — free, fast, and validates
        # the key in one call. See docs.dev.runwayml.com/api-details/.
        r = httpx.get(
            "https://api.dev.runwayml.com/v1/organization",
            headers={
                "Authorization": f"Bearer {key}",
                "X-Runway-Version": "2024-11-06",
            },
            timeout=10,
        )
        if r.status_code == 401:
            line("RUNWAYML_API_SECRET", "FAIL",
                 "invalid key (401) — re-issue at dev.runwayml.com")
            return False
        if r.status_code == 404:
            # Older versions of the API may not expose /organization;
            # fall back to listing tasks (also auth'd) to validate.
            r = httpx.get(
                "https://api.dev.runwayml.com/v1/tasks",
                headers={
                    "Authorization": f"Bearer {key}",
                    "X-Runway-Version": "2024-11-06",
                },
                timeout=10,
            )
            if r.status_code == 401:
                line("RUNWAYML_API_SECRET", "FAIL", "invalid key (401)")
                return False
            r.raise_for_status()
            line("RUNWAYML_API_SECRET", "OK", "key reachable")
            return True
        r.raise_for_status()
        data = r.json()
        # The response shape varies by API version; surface what we can.
        credits = data.get("creditBalance") or data.get("credits") or data.get("usageBalance")
        tier = data.get("tier") or data.get("usageTier")
        bits = []
        if credits is not None:
            # Runway credits = $0.01 each. A default 3-scene run is ~2,800 credits ($28).
            bits.append(f"{credits} credits (${credits / 100:.2f})")
        if tier:
            bits.append(f"tier {tier}")
        detail = ", ".join(bits) if bits else "key reachable"
        line("RUNWAYML_API_SECRET", "OK", detail)
        if credits is not None and credits < 600:
            line("RUNWAYML_API_SECRET", "WARN",
                 f"low balance — a default run costs ~2,800 credits ($28). Top up.")
        return True
    except Exception as e:  # noqa: BLE001
        msg = str(e).split("\n")[0][:120]
        line("RUNWAYML_API_SECRET", "FAIL", msg)
        return False


def check_google() -> bool:
    """Critic-only: Gemini 3 Pro for long-video review in evaluate_film.

    The agent can disable this and run with Claude-stills as the sole
    reviewer if they want. Optional rather than required.
    """
    key = os.getenv("GOOGLE_AI_API_KEY", "").strip()
    if not key or key.startswith("AIza..."):
        line("GOOGLE_AI_API_KEY", "SKIP",
             "optional — long-video critic disabled "
             "(Claude stills review still runs)")
        return True
    try:
        from google import genai  # type: ignore
        client = genai.Client(api_key=key)
        # 1-token reachability check, free under quota.
        client.models.generate_content(
            model="gemini-3.1-pro-preview",
            contents="ok",
            config={"max_output_tokens": 1},
        )
        line("GOOGLE_AI_API_KEY", "OK", "gemini-3.1-pro-preview reachable (critic enabled)")
        return True
    except ImportError:
        line("GOOGLE_AI_API_KEY", "FAIL",
             "google-genai not installed — run `uv sync`")
        return False
    except Exception as e:  # noqa: BLE001
        msg = str(e).split("\n")[0][:120]
        line("GOOGLE_AI_API_KEY", "FAIL", msg)
        return False


def check_stability() -> bool:
    key = os.getenv("STABILITY_API_KEY", "")
    if not key or key.startswith("sk-...") or key == "":
        line("STABILITY_API_KEY", "FAIL", "missing — see SETUP.md §4d")
        return False
    try:
        import httpx
        r = httpx.get(
            "https://api.stability.ai/v1/user/balance",
            headers={"Authorization": f"Bearer {key}"},
            timeout=10,
        )
        if r.status_code == 401:
            line("STABILITY_API_KEY", "FAIL", "invalid key (401)")
            return False
        r.raise_for_status()
        balance = r.json().get("credits", 0)
        # Stable Audio is roughly 10 credits per generation = $0.10.
        gens_left = int(balance / 10)
        if balance < 5:
            line("STABILITY_API_KEY", "WARN",
                 f"low balance: {balance:.1f} credits "
                 f"(~{gens_left} generations) — top up $10 minimum")
        else:
            line("STABILITY_API_KEY", "OK",
                 f"{balance:.1f} credits (~{gens_left} generations)")
        return True
    except Exception as e:  # noqa: BLE001
        msg = str(e).split("\n")[0][:120]
        line("STABILITY_API_KEY", "FAIL", msg)
        return False


# ---- system checks ------------------------------------------------------
def check_ffmpeg() -> bool:
    if not shutil.which("ffmpeg"):
        line("ffmpeg", "FAIL", "not on PATH — see SETUP.md §1")
        return False
    try:
        r = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True, timeout=5,
        )
        first = r.stdout.split("\n", 1)[0]
        # extract the version number; tolerate distro suffixes
        version = first.split(" ")[2] if len(first.split(" ")) > 2 else "?"
        major = int(version.split(".")[0]) if version[0].isdigit() else 0
        if major < 4:
            line("ffmpeg", "WARN",
                 f"version {version} — colorbalance filter may misbehave "
                 f"on builds older than 4.x")
        else:
            line("ffmpeg", "OK", f"version {version}")
        return True
    except Exception as e:  # noqa: BLE001
        line("ffmpeg", "FAIL", str(e)[:120])
        return False


def check_book() -> bool:
    path = Path(os.getenv(
        "BOOK_PDF_PATH",
        "/mnt/user-data/uploads/JurassicPark-MichaelCrichton.pdf",
    ))
    if not path.exists():
        line("book pdf", "FAIL",
             f"{path} not found — set BOOK_PDF_PATH in .env")
        return False
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            n = len(pdf.pages)
            sample = pdf.pages[0].extract_text() or ""
        if not sample.strip():
            line("book pdf", "WARN",
                 f"{path.name} ({n} pp) — page 1 has no extractable text "
                 f"(OCR'd scan?)")
            return True
        line("book pdf", "OK", f"{path.name} ({n} pp)")
        return True
    except Exception as e:  # noqa: BLE001
        line("book pdf", "FAIL", str(e)[:120])
        return False


# ---- main ---------------------------------------------------------------
def main() -> int:
    print()
    print("autofilm setup check")
    print("=" * 56)
    print()

    api_results = [
        check_anthropic(),
        check_runway(),
        check_google(),
        check_stability(),
    ]
    print()
    sys_results = [
        check_ffmpeg(),
        check_book(),
    ]
    print()

    ok = all(api_results + sys_results)
    if ok:
        print(f"  {GREEN}All systems go.{RESET} "
              f"`python produce.py` to run an experiment.")
        return 0
    elif all(api_results) and all(sys_results):
        # All passed but maybe with WARN — already printed above
        return 0
    else:
        n_fail = sum(1 for r in api_results + sys_results if not r)
        print(f"  {RED}{n_fail} check(s) failed.{RESET} "
              f"See SETUP.md for the section referenced above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
