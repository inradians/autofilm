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


def check_openai() -> bool:
    key = os.getenv("OPENAI_API_KEY", "")
    if not key or key.startswith("sk-proj-...") or key == "":
        line("OPENAI_API_KEY", "FAIL", "missing — see SETUP.md §4b")
        return False
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        models = list(client.models.list().data)
        ids = {m.id for m in models}
        # gpt-image-2 only appears for verified orgs.
        if "gpt-image-2" in ids:
            line("OPENAI_API_KEY", "OK",
                 f"org verified, {len(ids)} models incl. gpt-image-2")
            return True
        if "gpt-image-1" in ids:
            line("OPENAI_API_KEY", "WARN",
                 "key works but gpt-image-2 not visible — verify your org "
                 "(SETUP.md §4b step 3); gpt-image-1 is the older fallback")
            return True
        line("OPENAI_API_KEY", "WARN",
             f"key works but no gpt-image-* models visible — "
             f"org likely unverified (SETUP.md §4b step 3)")
        return True
    except Exception as e:  # noqa: BLE001
        msg = str(e).split("\n")[0][:120]
        line("OPENAI_API_KEY", "FAIL", msg)
        return False


def check_google() -> bool:
    key = os.getenv("GOOGLE_AI_API_KEY", "")
    if not key or key.startswith("AIza...") or key == "":
        line("GOOGLE_AI_API_KEY", "FAIL", "missing — see SETUP.md §4c")
        return False
    try:
        from google import genai  # type: ignore
        client = genai.Client(api_key=key)
        # A 1-token gemini call validates the key. Free under quota.
        resp = client.models.generate_content(
            model="gemini-3-pro",
            contents="ok",
            config={"max_output_tokens": 1},
        )
        # Probe Veo availability separately. The list_models call is free
        # and returns whatever models the project has access to. Veo only
        # shows up if billing is enabled.
        try:
            available = {m.name for m in client.models.list()}
            has_veo = any("veo-3.1" in n for n in available)
            if has_veo:
                line("GOOGLE_AI_API_KEY", "OK",
                     "billing enabled, veo-3.1 + gemini-3-pro available")
            else:
                line("GOOGLE_AI_API_KEY", "WARN",
                     "key works for Gemini but Veo 3.1 not listed — "
                     "enable billing (SETUP.md §4c step 5)")
        except Exception:  # noqa: BLE001
            # list_models can be flaky depending on project shape; fall
            # back to "key works" if we got the gemini reply.
            line("GOOGLE_AI_API_KEY", "OK", "gemini-3-pro reachable")
        return True
    except ImportError:
        line("GOOGLE_AI_API_KEY", "FAIL",
             "google-genai not installed — run `uv sync`")
        return False
    except Exception as e:  # noqa: BLE001
        msg = str(e).split("\n")[0][:120]
        line("GOOGLE_AI_API_KEY", "FAIL", msg)
        return False


def check_elevenlabs() -> bool:
    key = os.getenv("ELEVENLABS_API_KEY", "").strip()
    if not key or key.startswith("sk_..."):
        line("ELEVENLABS_API_KEY", "SKIP",
             "optional — ambient SFX layer disabled "
             "(Veo's native audio + music still cover scenes)")
        return True  # optional, never fails the check
    try:
        import httpx
        r = httpx.get(
            "https://api.elevenlabs.io/v1/user/subscription",
            headers={"xi-api-key": key},
            timeout=10,
        )
        if r.status_code == 401:
            line("ELEVENLABS_API_KEY", "FAIL", "invalid key (401)")
            return False
        r.raise_for_status()
        sub = r.json()
        tier = sub.get("tier", "unknown")
        used = sub.get("character_count", 0)
        cap = sub.get("character_limit", 0)
        left = max(cap - used, 0)
        line("ELEVENLABS_API_KEY", "OK",
             f"{tier} plan, {left:,} chars left")
        return True
    except Exception as e:  # noqa: BLE001
        msg = str(e).split("\n")[0][:120]
        line("ELEVENLABS_API_KEY", "FAIL", msg)
        return False


def check_stability() -> bool:
    key = os.getenv("STABILITY_API_KEY", "")
    if not key or key.startswith("sk-...") or key == "":
        line("STABILITY_API_KEY", "FAIL", "missing — see SETUP.md §4e")
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
        check_openai(),
        check_google(),
        check_elevenlabs(),
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
