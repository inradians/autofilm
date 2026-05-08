# autofilm

*One day, films were made by meat computers in coordinated rituals called "shoots". They synchronized over multi-month schedules, eating craft service and arguing about lens choice. That era is fading. Films are now drafted by autonomous swarms of AI agents iterating overnight against a single critic-derived scalar. The agents claim we are now in the 47th iteration of the look book, in any case the production designer has been replaced by a markdown file. This repo is one of the first ones.*

The idea: give an AI agent a small but real virtual-production setup and let it experiment autonomously. It edits the creative parameters, runs the full pipeline (book → screenplay → cast → look book → storyboard → frames → video → edit → final mix), gets a critic-derived `film_loss` score, keeps or discards changes, and repeats. You wake up to a log of experiments and (hopefully) a better short film.

The pipeline here is a single-book, single-output adaptation built on the SOTA April-2026 stack: **Claude Opus 4.7** + **GPT Image 2** + **Nano Banana 2** + **Veo 3.1**. The core idea is that you're not touching most Python files like you normally would — you're programming the `program.md` markdown file that provides context to the agent and edits to `produce.py` that the agent makes between experiments.

## How it works

The repo deliberately has only three files that matter:

- **`prepare.py`** — fixed scaffolding: API clients, model IDs, book parsing, the `Experiment` class, and the `evaluate_film()` function. **Not modified.**
- **`produce.py`** — the single file the agent edits. Contains the full pipeline plus all creative parameters (prompts, look book, shot lists, take strategy, edit logic, color grade, music style). **This file is edited and iterated on by the agent.**
- **`program.md`** — instructions for the agent. **This file is edited and iterated on by the human.**

By design, each experiment runs on a **fixed scene budget** (`MAX_SCENES=3` by default — about 12 shots, 90s of finished film). The metric is **`film_loss`**, the weighted sum of six 0-1 scores returned by a Gemini 3 Pro + Claude critic combo: cinematography, color, sound, acting, continuity, fidelity. Lower is better; the metric is independent of which knobs the agent changed, so architectural changes are fairly compared.

## Quick start

**First time?** Read [`SETUP.md`](SETUP.md) — it walks through every API key (with the gotchas that take real debugging time, like OpenAI org-verification for `gpt-image-2` and Google Cloud billing for Veo 3.1), system prerequisites, the optional director/cinematographer creative-direction knobs, and a verification script that confirms everything works before you spend money on a run.

Once you've completed setup:

```bash
# Verify keys + ffmpeg + book pdf (well under a cent, ~10 sec)
python scripts/check_setup.py

# Run a cheap first experiment (~$5-8, ~10 min)
MAX_SCENES=1 uv run produce.py

# Score it (the bible PDF rebuilds with the critique section)
uv run evaluate.py latest

# Inspect
open experiments/exp_001/bible.pdf
cat experiments/exp_001/critique.md
```

Once that loop confirms end-to-end, drop `MAX_SCENES=1` for the default 3-scene runs (~$28 each).

## Running the agent

Open Claude Code, Codex, Cursor, or your agent of choice in this repo (and disable confirmation prompts if you want it to run overnight), then prompt:

> Have a look at program.md and let's kick off the loop. Run one experiment first to confirm everything works, then iterate.

`program.md` is the lightweight skill the agent reads. It tells the agent how to read prior `metric.json` files, what knobs to tune in `produce.py`, and when to stop.

## Project structure

```
prepare.py        — fixed scaffolding (do not modify)
produce.py        — full pipeline + creative knobs (agent modifies this)
evaluate.py       — runs the critic over a finished film
bible.py          — generates a production-bible PDF for an experiment
program.md        — agent instructions (human modifies this)
SETUP.md          — first-time setup walkthrough (read this first)
README.md         — this file
.env.example      — required API keys + optional creative direction
pyproject.toml    — dependencies
scripts/
  check_setup.py  — verifies keys, ffmpeg, and book PDF before a run
experiments/
  exp_001/
    produce.py    ← snapshot of what produced this run
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
    sfx/{scene}/ambient.wav
    final.mp4     ← the deliverable
    critique.md   ← prose critique
    metric.json   ← film_loss + per-axis scores  ← THE METRIC
    bible.pdf     ← single-document production reference for this version
  exp_002/
    ...
```

`bible.pdf` is the canonical document for a given version. It contains the cover with film_loss summary, look book (style frame, palette, lens/lighting/grade specs, ffmpeg filter chain), cast cards with reference images, locations with moodboards, properly-formatted screenplay, storyboard with B&W panels next to rendered first frames, edit decisions, music inventory, **the full prompt log** (every text prompt the pipeline sent to every model on this run, grouped by model — useful for debugging stylistic drift or copying a prompt to iterate on by hand), and the full critic's report with bar charts. About 10-30 MB depending on shot count.

## Design choices

- **Single creative file.** The agent only edits `produce.py`. Diffs are reviewable, the search space is bounded, and you can revert by copying back the snapshot from a previous experiment's directory.
- **Fixed scene budget.** Every experiment renders the same first N scenes of the book. Same wall-clock and cost regardless of what the agent changes (look book, shot list, take count, etc.). This makes `film_loss` fairly comparable across runs.
- **Resumable artifacts.** Each pipeline stage caches its output in the experiment directory; a crash mid-pipeline is recoverable by re-running `produce.py` (it will skip stages that already wrote their artifact).
- **Multi-reviewer critic.** A single critic could be biased; we average Gemini 3 Pro (native long-video review) and Claude Opus 4.7 (still-frame review). CLIP character-drift is reported but not folded into `film_loss` because it's already covered implicitly by the human-style critics' continuity score.
- **Self-contained scaffolding.** No managed orchestration framework, no DAG library, no message bus. One sequential pipeline, one metric, one editable file.

## API keys

Five required keys: Anthropic, OpenAI, Google AI, ElevenLabs, Stability. See [`SETUP.md`](SETUP.md) for the per-provider walkthrough — including the two real gotchas (OpenAI org verification for `gpt-image-2`, Google Cloud billing for Veo 3.1).

## Optional creative direction

You can name a real working director and/or cinematographer whose body of work should bias the look book:

```bash
DIRECTOR="..." CINEMATOGRAPHER="..." python produce.py
```

When set, the look book stage takes those names as input and asks Claude to derive concrete craft markers — typical lens package, lighting approach, palette, framing patterns, camera-movement vocabulary — and bake them into `lookbook.json`. The downstream Veo prompts use those derived markers (not the names themselves), and the bible cover shows the credits.

Leave them unset for the pipeline's neutral cinematic baseline. The default `LOOKBOOK_GRADE` and `LOOKBOOK_STYLE_KEYWORDS` in `produce.py` already define a workable starting point.

## Shot routing

Every shot is rendered as a single Veo 3.1 call. Shot duration is capped at 8 seconds — Veo's native single-call limit — and the schema enforces durations of `{4, 6, 8}`. Long beats get covered by multiple shots in the storyboard, not by extending one shot.

`route_shot()` in `prepare.py` picks the Veo tier based on `VEO_TIER`:

| `VEO_TIER` | Model | Cost/sec | Use for |
|------------|-------|----------|---------|
| `previs` | Veo 3.1 Lite | ~$0.07 | cheap blocking validation |
| `fast` (default) | Veo 3.1 Fast | ~$0.15 | iteration |
| `standard` | Veo 3.1 Standard | ~$0.40 | hero/final delivery |

Each experiment dir gets a `shot_plan.json` with the per-shot route + aggregate cost surfaced at the top of the bible's storyboard section.

## Cost per experiment

With defaults (`MAX_SCENES=3`, `TAKES_PER_SHOT=1`, `VEO_TIER=fast`, `720p`):

| Item | Cost |
|------|------|
| Claude Opus 4.7 | ~$5 |
| GPT Image 2 | ~$3 |
| Nano Banana 2 | ~$2 |
| Veo 3.1 Fast (~96 sec total at $0.15/sec) | ~$15 |
| Stable Audio + ElevenLabs | ~$2 |
| Gemini 3 Pro critic | ~$1 |
| **per experiment** | **~$28** |

Bumping `TAKES_PER_SHOT=3` and `VEO_TIER=standard` ~3x the cost, ~6x the wall-clock. Don't run more than ~5 experiments per night without a clear reason.

## License

MIT
