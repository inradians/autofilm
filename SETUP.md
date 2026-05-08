# Setting up autofilm

This is the first-time setup guide for the **Runway-consolidated stack**. Compared to the original 5-provider setup, this version goes from five required keys to three, drops the two annoying approval delays (OpenAI org verification and Google Cloud billing), and gets you to a green setup-check in **about 15 minutes** for **~$25 in prepaid credits**.

## 1. Prerequisites

| Tool | Version | Why |
|------|---------|-----|
| **Python** | 3.10 or newer (3.11 recommended) | matches `pyproject.toml` |
| **ffmpeg** | 4.4+ | the color-grade chain uses `colorbalance` and `eq`; older builds choke |
| **uv** | latest | recommended for dep management; `pip` works as fallback |

Install ffmpeg:

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt update && sudo apt install -y ffmpeg

# Windows
winget install ffmpeg
```

Verify: `ffmpeg -version` should print 4.x or higher.

Install uv:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 2. Install the project

```bash
git clone <your-fork-or-zip>
cd autofilm
uv sync           # creates .venv and installs deps
source .venv/bin/activate  # or: . .venv/Scripts/activate on Windows
```

If you're on `pip` instead: `pip install -e .` will work but is slower.

## 3. Create your .env file

```bash
cp .env.example .env
```

Fill in the keys as you acquire them in step 4. The `.env` file is gitignored — never commit it.

## 4. Get the API keys

You need **three keys**, plus one optional. None has an approval delay this time, so you can do them in any order.

### 4a. Anthropic — Claude Opus 4.7

Powers script parsing, casting, look book generation, edit decisions, and the stills critic reviewer.

1. Go to **https://console.anthropic.com**
2. Sign in. Click **Settings → API Keys** in the left nav.
3. Click **Create Key**, name it (e.g. "autofilm"), copy the value (`sk-ant-...`).
4. Click **Plans & Billing → Add credits** and load at least $20. Anthropic is prepaid.
5. Paste into `.env`:

```
ANTHROPIC_API_KEY=sk-ant-api03-...
```

**Cost expectation:** ~$5 per default-size experiment for Claude calls.

---

### 4b. Runway — image, video, and SFX in one API

This single key replaces what used to be **four** separate provider integrations: OpenAI (GPT Image 2), Google AI (Nano Banana 2 + Veo 3.1), and ElevenLabs (SFX). Runway proxies all of those models — including Veo 3.1 — through one credit-billed API.

1. Go to **https://dev.runwayml.com/**
2. Sign in (or create an account). Click **API Keys** in the left nav.
3. Click **Create API Key**, name it (e.g. "autofilm"), copy the value (`key_...`).
4. Click **Usage & Billing** and add credits. Runway is prepaid at **$0.01 per credit**. Top up at least **$20 (2,000 credits)**. A default 3-scene experiment uses ~2,200 credits ($22).
5. Paste into `.env`:

```
RUNWAYML_API_SECRET=key_...
```

**Cost expectation:** ~$22 per default-size experiment, dominated by Veo at 15 credits/sec ($0.15/sec) for ~96 sec of generated video. Drop to ~$10 with `MAX_SCENES=1`.

**What you can pick from on this single key:**

| Model | Use | Cost |
|---|---|---|
| `veo3.1_fast` | default video model, native audio | 10–15 c/s |
| `veo3.1` | hero/final delivery, native audio | 20–40 c/s |
| `gen4.5` | image-to-video with **native reference-image identity-lock** | 12 c/s |
| `seedance2` | up to 15s in one call (busts the 8s Veo cap) | 36 c/s |
| `gen4_aleph` | video-to-video transformation (per-shot grading) | 15 c/s |
| `gpt_image_2` | first-frame composition (best instruction-following) | 1–41 c/img by quality |
| `gemini_image3_pro` | Nano Banana — identity refs, multi-image fusion | 20 c/img |
| `gen4_image` / `gen4_image_turbo` | Runway-native, strongest for character continuity | 5–8 c / 2 c |
| `eleven_text_to_sound_v2` | ambient SFX beds | 1 c/s |
| `eleven_multilingual_v2` | TTS narrator (new) | 1 c/50 chars |

**Region note:** Some Veo features have limited regional availability for `personGeneration` settings. The pipeline uses `allow_adult` mode which works in most regions; if you see a `personGeneration` error, file an issue.

---

### 4c. Stability — Stable Audio 2.5

Powers the music score per scene. Up to 47 seconds per generation, instrumental. **Kept on the direct Stability API because Runway has no music model** — its audio offerings are SFX and TTS only.

1. Go to **https://platform.stability.ai**
2. Sign up. Click **Account → API keys → Create API key**.
3. Click **Account → Billing → Add credits**. Minimum top-up is $10. Each generation costs ~$0.10.
4. Paste into `.env`:

```
STABILITY_API_KEY=sk-...
```

**Cost expectation:** ~$1 per experiment.

---

### 4d. Google AI — Gemini 3 Pro long-video critic (OPTIONAL)

This is now optional. Veo 3.1 has moved to Runway, so the Google AI key only powers **Reviewer A** in `evaluate_film()` — the long-video critic that watches the actual `final.mp4` end-to-end with timestamp citations. Without it, `evaluate.py` falls back to Reviewer B (Claude Opus 4.7 on representative stills) as the sole reviewer.

Skip it if: you're cost-sensitive, you trust Claude's stills review, you don't want to enable Google Cloud billing.

If you want the second reviewer:

1. Go to **https://aistudio.google.com**
2. Click **Get API key** (top right). Use any project — billing isn't required for Gemini 3 Pro under the free tier as long as you stay under quota.
3. Copy the key (`AIza...`).
4. Paste into `.env`:

```
GOOGLE_AI_API_KEY=AIza...
```

**Cost expectation:** ~$1 per experiment for the critic, or $0 if skipped.

**Why we keep this on direct Google AI:** Runway proxies many Google models (Veo, Gemini Image 3 Pro, Gemini 2.5 Flash) but not the long-video review path. The Gemini 3 Pro `generateContent` endpoint with a video file attachment + tool-calling response is the cleanest way to get scored axis output, and that path goes through the Google AI Studio API directly.

---

## 5. Optional creative direction

Two more env vars in `.env` let you point the look book at a real working director and/or cinematographer. When set, the look book stage derives concrete craft markers (lens choice, lighting approach, palette, framing patterns) from their published filmography and bakes those into `lookbook.json`. The bible cover shows them as credits.

```
DIRECTOR=Denis Villeneuve
CINEMATOGRAPHER=Roger Deakins
```

Both are optional and independent — set just one, both, or neither. Leave blank for the pipeline's neutral cinematic baseline.

## 6. Place the source book

The default `BOOK_PDF_PATH` points at `/mnt/user-data/uploads/JurassicPark-MichaelCrichton.pdf`. Override in `.env`:

```
BOOK_PDF_PATH=/absolute/path/to/your-book.pdf
```

Any plain-text-extractable PDF works. Scans without OCR don't.

## 7. Verify your setup

Run the included check before your first real experiment:

```bash
python scripts/check_setup.py
```

It pings each provider with a free or near-free call, reports OK / FAIL / SKIP per key, and warns about common issues. Total cost: a fraction of a cent. Total time: ~10 seconds.

A clean output looks like:

```
ANTHROPIC_API_KEY      OK   claude-opus-4-7 reachable
RUNWAYML_API_SECRET    OK   2000 credits ($20.00)
GOOGLE_AI_API_KEY      OK   gemini-3-pro reachable (critic enabled)
STABILITY_API_KEY      OK   180.0 credits (~18 generations)

ffmpeg                 OK   version 6.1.1
book pdf               OK   JurassicPark-MichaelCrichton.pdf (211 pp)

All systems go. python produce.py to run an experiment.
```

`SKIP` next to `GOOGLE_AI_API_KEY` is fine — that key is optional and the long-video critic just degrades to Claude-stills-only review. If any line says FAIL, the message tells you exactly which step in section 4 to revisit.

### 7b. Validate the Runway SDK migration (recommended before your first paid run)

The migration from OpenAI/Google AI/ElevenLabs to a Runway-consolidated stack changed the SDK parameter names and request body shapes for every image/video/SFX/TTS call. Static type-checking can't catch errors like `promptImage` vs `prompt_image` or `presetId` vs `preset_id` — those only surface against the live API.

`scripts/runway_smoke_test.py` exercises every Runway SDK call site in `prepare.py` with the cheapest possible inputs and reports OK/FAIL per call.

```bash
# Quick: image + audio only, ~$0.30, ~30 sec
RUNWAYML_API_SECRET=key_... python scripts/runway_smoke_test.py

# Full sweep including Veo + Aleph, ~$1.50, 3-5 min
RUNWAYML_API_SECRET=key_... python scripts/runway_smoke_test.py --include-video

# Just one helper:
python scripts/runway_smoke_test.py --only veo
```

When something fails the FAIL line gives the prepare.py line range to inspect plus the full Runway error. Skip this if you're confident the SDK calls are right — but at least once before running an actual `produce.py` experiment, it's a cheap insurance policy against losing $25 to a typo.

## 8. First run — cost-aware

For your very first experiment, cap everything tight:

```bash
MAX_SCENES=1 TAKES_PER_SHOT=1 VEO_TIER=fast VEO_RESOLUTION=720p \
  uv run produce.py
```

That generates one scene (~3-6 shots, ~24-48 sec of video) for **~$5-7** instead of the default ~$22. You'll get back an `experiments/exp_001/` directory with everything: cast, locations, look book, storyboard, references, frames, clips, music, EDL, `final.mp4`, and `bible.pdf`. Open the bible to see the full output — including a "Prompts" section that shows every prompt the pipeline sent to every model on this run, organized by model. Useful for understanding what the agent actually did.

Once you've confirmed it works end-to-end, raise `MAX_SCENES` to 3 (default) or higher.

To score the result and produce a critic's report:

```bash
uv run evaluate.py latest
```

This refreshes the bible PDF with the critique section and writes `metric.json` with the `film_loss` score plus structured `changes` suggestions.

## 9. Cost summary

This is a self-funded weekend project — every dollar is yours. Two cost shapes worth budgeting against:

**Minimum viable run** (`MAX_SCENES=1`, ambient SFX off, Veo Fast, 720p, Claude-stills-only review) — about **$6 per experiment**:

| Provider | Cost |
|----------|------|
| Anthropic | ~$2 |
| Runway (gpt_image_2 + nano-banana + Veo Fast 24-48s) | ~$3 |
| Stability | <$1 |
| **per experiment** | **~$6** |

**Default run** (`MAX_SCENES=3`, ambient SFX off, Veo Fast, 720p, both reviewers) — about **$28 per experiment**:

| Provider | Cost |
|----------|------|
| Anthropic (Claude Opus 4.7) | ~$5 |
| Runway: image (gpt_image_2 + nano-banana) | ~$5 |
| Runway: video (Veo 3.1 Fast — ~96s @ $0.15/s) | ~$15 |
| Runway: SFX (skipped by default) | ~$0 |
| Stability (Stable Audio) | ~$1 |
| Google AI (Gemini 3 Pro critic) | ~$1 |
| **per experiment** | **~$27** |

`TAKES_PER_SHOT=3` roughly triples the run cost. `VEO_TIER=standard` raises Veo's cost from $0.15/sec to $0.40/sec. `AMBIENT_SFX_ENABLED=1` adds ~$1/run.

A reasonable pattern: do a single MAX_SCENES=1 smoke test (~$6) to verify the pipeline works on your keys, then 2-3 default experiments (~$27 each) over a weekend. Total commitment: $60-90.

## 10. Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `RUNWAYML_API_SECRET is not set` | key missing from environment | Section 4b — `cp .env.example .env` and fill in |
| `401 Unauthorized` (Runway) | invalid key, or revoked | Reissue at dev.runwayml.com → API Keys |
| `Task failed — SAFETY.INPUT.*` (Runway) | content moderation rejected the prompt | Try a less violent or less explicit prompt |
| `Task failed — ASSET.INVALID` (Runway) | image/video format not supported | Re-encode as PNG/JPEG (image) or H.264 MP4 (video) |
| `429 Rate limited` (Runway) | hit the per-tier rate limit | Wait — `prepare.py` retries with backoff |
| `insufficient credits` (Runway) | run out of credits mid-experiment | Top up at dev.runwayml.com → Billing |
| `insufficient credits` (Stability) | account out of credits | Top up $10 minimum |
| Bible has empty frame slots | image task failed mid-pipeline | Re-run `produce.py` — stages are idempotent |
| Bible has empty video clips | Veo task failed | Check Runway dashboard for the failed task id |
| `No module named 'runwayml'` | dep not installed | `uv sync` to refresh |
| `Unknown encoder libx264` (ffmpeg) | stripped ffmpeg build | Reinstall full ffmpeg from official source |
| `pdfplumber: No /Pages` | encrypted or scanned PDF | Use a different copy or OCR it first |

If a stage keeps failing for environmental reasons (network, API errors, timeouts), don't iterate around it — the pipeline is supposed to crash loudly on infrastructure problems so you can fix them at the source.

## 11. What's where

```
autofilm/
├── prepare.py        # fixed scaffolding: API clients, Runway/Claude/Gemini helpers, Experiment class
├── produce.py        # the file the agent edits: prompts, look book, shot list, etc.
├── evaluate.py       # runs the critic, writes metric.json
├── bible.py          # auto-builds bible.pdf from experiment artifacts
├── program.md        # instructions to the autonomous agent
├── README.md         # project overview
├── SETUP.md          # this file
├── CHANGELOG.md      # migration history
├── .env.example      # template for your .env
├── pyproject.toml    # dependencies
├── scripts/
│   └── check_setup.py  # ping each provider, verify everything works
└── .claude/
    └── skills/         # vendored Runway agent skills (auto-discovered by Claude Code)
        ├── rw-generate-video/
        ├── rw-generate-image/
        └── rw-generate-audio/
```

Once you've made it through this guide once, day-to-day use is just: edit `produce.py`, run `python produce.py`, run `python evaluate.py latest`, open `experiments/exp_NNN/bible.pdf`, repeat.
