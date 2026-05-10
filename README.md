# autofilm

*One day, films were made by meat machines using bulky equipment in coordinated rituals called "shoots". They synchronized over multi-month schedules, eating craft service and arguing about lens choice. That era is fading. Films are now drafted by autonomous swarms of AI agents iterating overnight against a single critic-derived scalar. The agents claim we are now in the 47th iteration of the look book; in any case the production designer has been replaced by a markdown file. This repo is one of the first ones.*

The idea: give an AI agent a small but real virtual-production setup and let it experiment autonomously. It edits the creative parameters, runs the full pipeline (book → screenplay → cast → look book → storyboard → frames → video → edit → final mix), gets a critic-derived `film_loss` score, keeps or discards changes, and repeats. You wake up to a log of experiments and (hopefully) a better short film.

The pipeline runs on the SOTA May-2026 stack, **consolidated through Runway**: a single Runway API key replaces what used to be four separate provider integrations (OpenAI for GPT Image 2, Google AI for Nano Banana 2 + Veo 3.1, ElevenLabs for SFX). Anthropic for Claude Opus 4.7 and Stability for Stable Audio 2.5 are kept on direct APIs. Total: **3 keys instead of 5**, **0 approval delays instead of 2** (no more OpenAI org verification or Google Cloud billing for video).

## How it works

The repo deliberately has only three files that matter:

- **`prepare.py`** — fixed scaffolding: API clients, model IDs, book parsing, the `Experiment` class, and the `evaluate_film()` function. **Not modified.**
- **`produce.py`** — the single file the agent edits. Contains the full pipeline plus all creative parameters (prompts, look book, shot lists, take strategy, edit logic, color grade, music style). **This file is edited and iterated on by the agent.**
- **`program.md`** — instructions for the agent. **This file is edited and iterated on by the human.**

By design, each experiment runs on a **fixed scene budget** (`MAX_SCENES=3` by default — about 12 shots, 90s of finished film). The metric is **`film_loss`**, the weighted sum of six 0-1 scores returned by a Gemini 3 Pro + Claude critic combo: cinematography, color, sound, acting, continuity, fidelity. Lower is better; the metric is independent of which knobs the agent changed, so architectural changes are fairly compared.

## Quick start

**First time?** Read [`SETUP.md`](SETUP.md) — three keys, no approval delays, ~15 minutes to a green setup-check.

Once you've completed setup:

```bash
# Verify keys + ffmpeg + book pdf (well under a cent, ~10 sec)
python scripts/check_setup.py

# Run a cheap first experiment (~$5-7, ~10 min)
MAX_SCENES=1 uv run produce.py

# Score it (the bible PDF rebuilds with the critique section)
uv run evaluate.py latest

# Inspect (paths are now per-book: experiments/{book_slug}/exp_NNN/)
open experiments/jurassic_park/exp_001/bible.pdf
cat experiments/jurassic_park/exp_001/critique.md
```

Once that loop confirms end-to-end, drop `MAX_SCENES=1` for the default 3-scene runs (~$27 each).

## Test books

The pipeline targets a single source PDF at a time, set via `BOOK_PDF_PATH` in `.env`. Three deliberately-different books in the test set, each chosen to stress a different axis of the pipeline:

| Book | Author, year | Why it's in the set | Wikipedia |
|---|---|---|---|
| **Jurassic Park** *(default)* | Michael Crichton, 1991 | Dialogue-heavy thriller with discrete locations, a manageable cast, and clean narrative beats — the easy mode of the three. Stresses **continuity** (re-use of locations and characters across scenes) and **fidelity** (the book's set-pieces are well-known, so deviations are obvious). | [link](https://en.wikipedia.org/wiki/Jurassic_Park) |
| **Last Exit to Brooklyn** | Hubert Selby Jr., 1964 | Episodic, transgressive, low-incident, written without conventional punctuation. Stresses **acting** (much of the book is interior monologue rendered as fragmented dialogue) and **cinematography** (locations are bleak and similar — the look book has to do real work). |  [link](https://en.wikipedia.org/wiki/Last_Exit_to_Brooklyn) |
| **The Electric Kool-Aid Acid Test** | Tom Wolfe, 1968 | New Journalism, non-linear, drug-addled, second-person digressions. Stresses **fidelity** (Wolfe's voice is the book — losing it loses everything) and **sound** (the music cue and ambient layer carry a lot of the period feel). | [link](https://en.wikipedia.org/wiki/The_Electric_Kool-Aid_Acid_Test) |

**We will use Jurassic Park first.** It's already the default `BOOK_PDF_PATH`. Drop the PDF at `/mnt/user-data/uploads/JurassicPark-MichaelCrichton.pdf` (or override the env var) and you're ready to run. The other two are for the second and third experiment runs once the pipeline is dialed in.

## Running the agent

Open Claude Code, Codex, Cursor, or your agent of choice in this repo (and disable confirmation prompts if you want it to run overnight), then prompt:

> Have a look at program.md and let's kick off the loop. Run one experiment first to confirm everything works, then iterate.

`program.md` is the lightweight skill the agent reads. It tells the agent how to read prior `metric.json` files, what knobs to tune in `produce.py`, when to switch between Runway video models, and when to stop. The `.claude/skills/` directory ships vendored Runway skills (`rw-generate-video`, `rw-generate-image`, `rw-generate-audio`) that Claude Code auto-discovers — useful when you want the agent to make one-off generations outside the main pipeline.

## The stack

| Stage | Model | Provider | Cost (default run) |
|---|---|---|---|
| Script parsing, casting, look book, edit decisions | Claude Opus 4.7 | Anthropic | ~$5 |
| First-frame composition | `gpt_image_2` | Runway | ~$2 |
| Identity-lock / character refs | `gemini_image3_pro` (Nano Banana) | Runway | ~$2 |
| Per-shot video generation | `veo3.1_fast` | Runway | ~$15 |
| Music score | Stable Audio 2.5 | Stability | ~$1 |
| Ambient SFX (off by default) | `eleven_text_to_sound_v2` | Runway | ~$1 |
| Long-video critic | Gemini 3 Pro | Google AI (optional) | ~$1 |
| Stills critic | Claude Opus 4.7 | Anthropic | (rolled into Anthropic line) |
| **per default experiment** | | | **~$27** |

**Alternative video models the agent can switch into:**

- `gen4.5` (12 c/s, $0.12/s) — Runway flagship with **native reference-image support**. Cheaper than Veo Fast and stronger at character continuity. Trade-off: no native dialogue audio, so spoken scenes need TTS layered in.
- `seedance2` (36 c/s, $0.36/s) — supports up to **15 seconds in a single call**, busting the 8-second Veo cap. Use sparingly for long beats.
- `gen4_aleph` (15 c/s, $0.15/s) — **video-to-video transformation**. Apply per-shot color/mood/seasonal grading on top of generated clips when the global ffmpeg `LOOKBOOK_GRADE` chain isn't enough.
- `veo3.1` (40 c/s with audio, $0.40/s) — hero-quality delivery tier.

Switch via `VEO_TIER=fast | standard | gen4.5 | seedance2 | previs` in `.env`, or per-shot inside `produce.py` by calling `route_shot(duration, tier="gen4.5")`.

## Project structure

Generated runs write under `experiments/{book_slug}/exp_NNN/`. That directory is **gitignored** so clones stay small; paths below describe what appears on disk after you run the pipeline.

```
prepare.py        — fixed scaffolding (do not modify)
produce.py        — full pipeline + creative knobs (agent modifies this)
evaluate.py       — runs the critic over a finished film
bible.py          — generates a production-bible PDF for an experiment
program.md        — agent instructions (human modifies this)
SETUP.md          — first-time setup walkthrough (read this first)
CHANGELOG.md      — migration history
README.md         — this file
.env.example      — required API keys + optional creative direction
pyproject.toml    — dependencies
scripts/
  check_setup.py  — verifies keys, ffmpeg, and book PDF before a run
.claude/skills/   — vendored Runway skills (auto-discovered by Claude Code)
experiments/
  jurassic_park/
    exp_001/
      produce.py    ← snapshot of what produced this run
      book.txt      ← book slug ("jurassic_park")
      script.json
      cast.json
      locations.json
      lookbook.json
      storyboard.json
      shot_plan.json
      frames/{scene}/{shot}.png
      clips/{scene}/{shot}/take_N.mp4
      edl.json
      music/{scene}.wav
      sfx/{scene}/ambient.wav    ← only if AMBIENT_SFX_ENABLED=1
      final.mp4     ← the deliverable
      critique.md   ← prose critique
      metric.json   ← film_loss + per-axis scores  ← THE METRIC
      bible.pdf     ← single-document production reference for this version
    exp_002/
      ...
  last_exit_to_brooklyn/
    exp_001/
      ...
  _smoke_tests/     ← Runway SDK validation outputs (scripts/runway_smoke_test.py)
    20260508_214500/
      gpt_image.png, veo.mp4, summary.md
```

`bible.pdf` is the canonical document for a given version. It contains the cover with film_loss summary, look book (style frame, palette, lens/lighting/grade specs, ffmpeg filter chain), cast cards with reference images, locations with moodboards, properly-formatted screenplay, storyboard with B&W panels next to rendered first frames, edit decisions, music inventory, **the full prompt log** (every text prompt the pipeline sent to every model on this run, grouped by model — useful for debugging stylistic drift or copying a prompt to iterate on by hand), and the full critic's report with bar charts. About 10–30 MB depending on shot count.

## Design choices

- **Single creative file.** The agent only edits `produce.py`. Diffs are reviewable, the search space is bounded, and you can revert by copying back the snapshot from a previous experiment's directory.
- **Fixed scene budget.** Every experiment renders the same first N scenes of the book. Same wall-clock and cost regardless of what the agent changes (look book, shot list, take count, etc.). This makes `film_loss` fairly comparable across runs.
- **Resumable artifacts.** Each pipeline stage caches its output in the experiment directory; a crash mid-pipeline is recoverable by re-running `produce.py` (it will skip stages that already wrote their artifact).
- **Multi-reviewer critic, optional.** A single critic could be biased; we average Gemini 3 Pro (native long-video review) and Claude Opus 4.7 (still-frame review). Gemini is now optional — `evaluate.py` degrades gracefully to Claude-only if `GOOGLE_AI_API_KEY` is unset. CLIP character-drift is reported but not folded into `film_loss` because it's already covered implicitly by the human-style critics' continuity score.
- **Self-contained scaffolding.** No managed orchestration framework, no DAG library, no message bus. One sequential pipeline, one metric, one editable file.
- **One billing surface for media.** Everything image/video/SFX runs through Runway credits. The agent doesn't have to reason about quota across four different vendor dashboards.

## API keys

Three required, one optional:

1. **`ANTHROPIC_API_KEY`** — Claude Opus 4.7
2. **`RUNWAYML_API_SECRET`** — image, video, and SFX
3. **`STABILITY_API_KEY`** — Stable Audio 2.5
4. **`GOOGLE_AI_API_KEY`** — *optional*, only for the long-video critic

See [`SETUP.md`](SETUP.md) for the per-provider walkthrough.

## Optional creative direction

You can name a real working director and/or cinematographer whose body of work should bias the look book:

```bash
DIRECTOR="..." CINEMATOGRAPHER="..." python produce.py
```

When set, the look book stage takes those names as input and asks Claude to derive concrete craft markers — typical lens package, lighting approach, palette, framing patterns, camera-movement vocabulary — and bake them into `lookbook.json`. The downstream Veo prompts use those derived markers (not the names themselves), and the bible cover shows the credits.

Leave them unset for the pipeline's neutral cinematic baseline. The default `LOOKBOOK_GRADE` and `LOOKBOOK_STYLE_KEYWORDS` in `produce.py` already define a workable starting point.

## Shot routing

Every shot is rendered as a single Runway video call. Default duration is capped at 8 seconds (Veo's native single-call limit, schema enforces `{4, 6, 8}`). Long beats get covered by multiple shots in the storyboard, not by extending one shot — *unless* the agent chooses `seedance2`, which lifts the cap to 15s.

`route_shot()` in `prepare.py` picks the model based on `VEO_TIER`:

| `VEO_TIER` | Model | Cost/sec (USD) | Use for |
|------------|-------|----------|---------|
| `previs` | `veo3.1_fast` | $0.15 | cheap blocking validation |
| `fast` (default) | `veo3.1_fast` | $0.15 | iteration |
| `standard` | `veo3.1` | $0.40 | hero/final delivery |
| `gen4.5` | `gen4.5` | $0.12 | identity-lock via reference images |
| `seedance2` | `seedance2` | $0.36 | long beats up to 15s |

Each experiment dir gets a `shot_plan.json` with the per-shot route + aggregate cost surfaced at the top of the bible's storyboard section.

## Cost per experiment

With defaults (`MAX_SCENES=3`, `TAKES_PER_SHOT=1`, `VEO_TIER=fast`, `720p`, ambient SFX off, Gemini critic on):

| Item | Cost |
|------|------|
| Claude Opus 4.7 | ~$5 |
| Runway: image generation | ~$5 |
| Runway: Veo Fast (~96 sec @ $0.15/sec) | ~$15 |
| Stable Audio | ~$1 |
| Gemini 3 Pro critic | ~$1 |
| **per experiment** | **~$27** |

Bumping `TAKES_PER_SHOT=3` and `VEO_TIER=standard` ~3× the cost, ~6× the wall-clock. Don't run more than ~5 experiments per night without a clear reason.

## License

MIT
