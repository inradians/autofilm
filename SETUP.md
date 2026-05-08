# Setting up autofilm

This is a complete first-time setup guide. Budget about 30 minutes the first time, mostly waiting for org-verification approvals on a couple of the providers. Once you have keys, you can run an experiment in ~20 minutes for ~$25.

## 1. Prerequisites

| Tool | Version | Why |
|------|---------|-----|
| **Python** | 3.10 or newer (3.11 recommended) | matches `pyproject.toml` |
| **ffmpeg** | 4.4+ | color grade chain uses `colorbalance` and `eq`; older builds choke |
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

You need four keys, plus one optional. Two of the required ones have approval steps that take 10-30 minutes; start those first so you can fill in the others while you wait.

### 4a. Anthropic — Claude Opus 4.7

Powers script parsing, casting, look book generation, edit decisions, and one of the two critic reviewers.

1. Go to **https://console.anthropic.com**
2. Sign in. Click **Settings → API Keys** in the left nav.
3. Click **Create Key**, name it (e.g. "autofilm"), copy the value (`sk-ant-...`).
4. Click **Plans & Billing → Add credits** and load at least $20. Anthropic is prepaid.
5. Paste into `.env`:

```
ANTHROPIC_API_KEY=sk-ant-api03-...
```

**Cost expectation:** ~$5/experiment for Claude calls.

---

### 4b. OpenAI — GPT Image 2

Powers first-frame composition and the look book's style frame. **Has the biggest setup gotcha in the whole stack — start this first.**

1. Go to **https://platform.openai.com**
2. Sign in. Click **Settings (gear icon) → Organization → General**.
3. Click **Verify Organization**. You'll be asked to upload a government ID. **This is required for `gpt-image-2`** — without verification, every image call returns HTTP 403 with a "must verify your organization" error message. Approval usually takes 5-30 minutes.
4. While you wait, click **API keys** in the left nav and **Create new secret key**. Copy `sk-proj-...`.
5. Click **Settings → Billing** and add at least $20. OpenAI is also prepaid.
6. Paste into `.env`:

```
OPENAI_API_KEY=sk-proj-...
```

**Cost expectation:** ~$3/experiment for image generations.

**If you skip verification:** every shot's first-frame call will fail with a 403 and the bible will have empty frame slots. The error message names the cause directly.

---

### 4c. Google AI — Nano Banana 2 + Veo 3.1 + Gemini 3 Pro

A single key drives three models: image generation (Nano Banana 2 for character refs and location moodboards), video generation (Veo 3.1 for every shot), and the long-video critic (Gemini 3 Pro). **Veo 3.1 is paid-only** — it is not on the Gemini free tier.

1. Go to **https://aistudio.google.com**
2. Sign in with a Google account.
3. Click **Get API key** in the top right. Either create a new project or use an existing Google Cloud project.
4. Copy the key (`AIza...`).
5. Critical: click **Set up billing** on the same page (or go to **https://console.cloud.google.com/billing** for the project). Link a billing account. **Veo 3.1 will refuse to run without billing enabled** — you'll get a "model not available for your project" error.
6. Paste into `.env`:

```
GOOGLE_AI_API_KEY=AIza...
```

**Cost expectation:** Veo 3.1 Fast at $0.15/sec dominates the bill — ~$15/experiment for ~96 sec of video. Nano Banana 2 and Gemini 3 Pro together add ~$3.

**Region note:** Veo has limited regional availability for `personGeneration` settings in EU/UK/Switzerland/MENA. The pipeline already uses `allow_adult` mode which works in those regions, but if you see a `personGeneration` error, that's why.

---

### 4d. ElevenLabs — Ambient SFX beds (OPTIONAL)

This one is optional. ElevenLabs adds an ambient sound bed per scene (rain, jungle, room tone) layered under Veo's native dialogue audio and the music cue. **You can skip this entirely** — Veo's synced audio plus the Stable Audio music already give you a complete scene mix, and the pipeline detects an empty `ELEVENLABS_API_KEY` and skips the SFX layer cleanly.

Skip it if: you're doing your first one-or-two runs, you want to keep monthly costs at zero, or you don't care about ambient texture under dialogue. Add it later once everything else is dialed in.

If you do want the ambient layer:

1. Go to **https://elevenlabs.io**
2. Sign up. Click your **Profile picture (top right) → Profile + API key**.
3. Copy the key.
4. Pick a tier on the **Subscription** page. The free tier has limited monthly quota that's enough for a few experiments. Paid tiers (Starter $5/mo, Creator $22/mo) raise the quota; cancel any time. ElevenLabs is the only provider in the stack billed by subscription rather than pay-as-you-go.
5. Paste into `.env`:

```
ELEVENLABS_API_KEY=sk_...
```

**Cost expectation:** $0/experiment if skipped, otherwise included in your subscription tier's monthly quota.

---

### 4e. Stability AI — Stable Audio 2.5

Powers the music score per scene. Up to 47 seconds per generation, instrumental.

1. Go to **https://platform.stability.ai**
2. Sign up. Click **Account → API keys → Create API key**.
3. Click **Account → Billing → Add credits**. Minimum top-up is $10. Each generation costs ~$0.10.
4. Paste into `.env`:

```
STABILITY_API_KEY=sk-...
```

**Cost expectation:** ~$1/experiment.

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

It pings each provider with the cheapest possible call (or a free metadata endpoint where available), reports OK/FAIL per key, and warns about the common gotchas. Total cost: a fraction of a cent. Total time: ~10 seconds.

A clean output looks like:

```
ANTHROPIC_API_KEY     OK   claude-opus-4-7 reachable
OPENAI_API_KEY        OK   org verified, gpt-image-2 listed
GOOGLE_AI_API_KEY     OK   billing enabled, veo-3.1-fast available
STABILITY_API_KEY     OK   credits: $18.40
ELEVENLABS_API_KEY    SKIP optional — ambient SFX layer disabled

ffmpeg                OK   version 6.1.1
book pdf              OK   /mnt/user-data/uploads/JurassicPark-MichaelCrichton.pdf (211 pp)

All systems go. python produce.py to run an experiment.
```

`SKIP` is the recommended default for ElevenLabs unless you specifically want the ambient sound bed. If any line says FAIL, the message tells you exactly which step in section 4 to revisit.

## 8. First run — cost-aware

For your very first experiment, cap everything tight:

```bash
MAX_SCENES=1 TAKES_PER_SHOT=1 VEO_TIER=fast VEO_RESOLUTION=720p \
  python produce.py
```

That generates one scene (~3-6 shots, ~24-48 sec of video) for **~$5-8** instead of the default ~$28. You'll get back an `experiments/exp_001/` directory with everything: cast, locations, look book, storyboard, references, frames, clips, music, EDL, `final.mp4`, and `bible.pdf`. Open the bible to see the full output — including a "Prompts" section that shows every prompt the pipeline sent to every model on this run, organized by model. Useful for understanding what the agent actually did.

Once you've confirmed it works end-to-end, raise `MAX_SCENES` to 3 (default) or higher.

To score the result and produce a critic's report:

```bash
python evaluate.py latest
```

This refreshes the bible PDF with the critique section and writes `metric.json` with the `film_loss` score plus structured `changes` suggestions.

## 9. Cost summary

This is a self-funded weekend project — every dollar is yours. Two cost shapes worth budgeting against:

**Minimum viable run** (`MAX_SCENES=1`, ElevenLabs skipped, Veo Fast, 720p) — about **$6 per experiment**:

| Provider | Cost |
|----------|------|
| Anthropic | ~$2 |
| OpenAI | ~$1 |
| Google (Nano Banana 2 + Veo Fast 24-48s + Gemini critic) | ~$3 |
| Stability | <$1 |
| **per experiment** | **~$6** |

**Default run** (`MAX_SCENES=3`, ElevenLabs on, Veo Fast, 720p) — about **$28 per experiment**:

| Provider | Cost |
|----------|------|
| Anthropic (Claude Opus 4.7) | ~$5 |
| OpenAI (GPT Image 2) | ~$3 |
| Google (Nano Banana 2) | ~$2 |
| Google (Veo 3.1 Fast — ~96 sec at $0.15/sec) | ~$15 |
| Stability (Stable Audio) | ~$1 |
| ElevenLabs (SFX) | ~$1 (or $0 if skipped) |
| Google (Gemini 3 Pro critic) | ~$1 |
| **per experiment** | **~$28** |

`TAKES_PER_SHOT=3` roughly triples the run cost. `VEO_TIER=standard` raises Veo's cost from $0.15/sec to $0.40/sec.

A reasonable pattern: do a single MAX_SCENES=1 smoke test (~$6) to verify the pipeline works on your keys, then 2-3 default experiments (~$28 each) over a weekend. Total commitment: $60-100. Cancel ElevenLabs and let prepaid credits sit otherwise.

## 10. Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `403 ... must verify your organization` (OpenAI) | Account not verified for image gen | Section 4b step 3 |
| `model not available for your project` (Veo) | Billing not enabled | Section 4c step 5 |
| `Ambient {scene} failed: ...` | ElevenLabs key invalid or quota hit | Either fix the key or unset `ELEVENLABS_API_KEY` to skip SFX entirely |
| `insufficient credits` (Stability) | Account out of credits | Top up $10 minimum |
| `PermissionDenied: personGeneration` (Veo) | EU/UK/etc region restriction | Already handled — file an issue if it persists |
| `No module named 'google.genai'` | Old `google-generativeai` package, not the new SDK | `uv sync` to refresh |
| `Unknown encoder libx264` (ffmpeg) | Stripped ffmpeg build | Reinstall full ffmpeg from official source |
| `pdfplumber: No /Pages` | Encrypted or scanned PDF | Use a different copy or OCR it first |
| Bible has empty frame slots | OpenAI 403 (org verification) | See first row above |
| Bible has empty video clips | Veo billing or region issue | See section 4c |

If a stage keeps failing for environmental reasons (network, API errors, timeouts), don't iterate around it — the pipeline is supposed to crash loudly on infrastructure problems so you can fix them at the source.

## 11. What's where

```
autofilm/
├── prepare.py        # fixed scaffolding: API clients, Veo/Claude/etc helpers, Experiment class
├── produce.py        # the file the agent edits: prompts, look book, shot list, etc.
├── evaluate.py       # runs the critic, writes metric.json
├── bible.py          # auto-builds bible.pdf from experiment artifacts
├── program.md        # instructions to the autonomous agent
├── README.md         # project overview
├── SETUP.md          # this file
├── .env.example      # template for your .env
├── pyproject.toml    # dependencies
└── scripts/
    └── check_setup.py  # ping each provider, verify everything works
```

Once you've made it through this guide once, day-to-day use is just: edit `produce.py`, run `python produce.py`, run `python evaluate.py latest`, open `experiments/exp_NNN/bible.pdf`, repeat.
