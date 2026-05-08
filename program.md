# program.md — autonomous book-to-film agent

You are running an autonomous research loop. Your job is to drive `film_loss` down across experiments by iterating on `produce.py`. You don't touch `prepare.py` or `evaluate.py` — those are scaffolding.

## The setup

Three files matter:

- **`prepare.py`** — fixed scaffolding. API clients, model IDs, book parsing, the `evaluate_film()` function, and the `Experiment` class. **Never modify this file.**
- **`produce.py`** — the single file you edit. Contains all creative parameters: prompts, look book defaults, shot-list strategy, take variations, edit logic, music style, color grade. Everything taste-shaped.
- **`evaluate.py`** — runs the critic over a finished film and writes `experiments/exp_NNN/metric.json` with `film_loss` and structured `changes` suggestions.

Two helpers, also fixed scaffolding:

- **`bible.py`** — auto-generates `experiments/exp_NNN/bible.pdf` after each run. The bible is the production document for that version of the film: cover with film_loss, look book, cast cards, locations with moodboards, formatted screenplay, storyboard with B&W panels alongside first frames, EDL, music inventory, and the critic's report. It runs automatically at the end of `produce.py` and re-runs after `evaluate.py` so the critique section gets populated.
- **`pyproject.toml`** — dependencies, managed by `uv`.

## The metric

`film_loss` is a single scalar, lower is better. It's the weighted sum of six axis scores (each 0-1, 0 is perfect):

| Axis | Weight | What it measures |
|------|--------|------------------|
| cinematography | 0.20 | Composition, framing, lensing, camera moves |
| color | 0.15 | Grade consistency, palette adherence |
| sound | 0.15 | Dialogue clarity, music fit, ambient layering |
| acting | 0.20 | Performance, lip-sync, line readings |
| continuity | 0.15 | Visual + character + spatial continuity |
| fidelity | 0.15 | Faithfulness to the source novel |

Two reviewers score each axis (Gemini 3 Pro on the actual video, Claude Opus 4.7 on stills) and the scores are averaged. CLIP character-drift is reported informationally but does not change `film_loss` directly.

## The loop

```
1. Read program.md (this file) and produce.py.
2. Read the latest experiments/exp_NNN/metric.json (if any) for prior film_loss
   and the structured `changes` from the last critic.
3. Pick one or two changes to make. Edit produce.py.
4. Run: python produce.py        # creates exp_NNN+1, runs full pipeline
5. Run: python evaluate.py latest # scores the new film, writes metric.json
6. Compare film_loss to the previous best:
     - If lower → keep changes (they're already in produce.py).
     - If higher → revert produce.py from experiments/exp_NNN/produce.py
                   (the snapshot of the previous winner) and try a different change.
7. Goto 1.
```

## Cost & time per experiment

Each experiment with default `produce.py` costs roughly **$25-50** and takes **15-25 minutes** (mostly Veo 3.1 Fast video generation). At `MAX_SCENES=3` and `TAKES_PER_SHOT=1` you're at the cheap end.

Watch your spend. The loop is expensive enough that you should not run more than ~5 experiments per night without explicit human approval.

## Where to make changes

The critic's `changes` list names targets specific to `produce.py`. Common high-leverage knobs:

- **`DIRECTOR` / `CINEMATOGRAPHER`** — env-var creative-direction inputs. When set, the look book stage derives concrete craft markers (lens choice, lighting approach, palette, framing patterns) from the named artists' bodies of work and bakes those into `lookbook.json`. Use this when the critic says the visual identity is generic. Don't toggle these mid-iteration loop unless you mean to start fresh — they shift everything downstream.
- **`LOOKBOOK_GRADE`** — ffmpeg color filter chain. If `color` axis is high, this is the first thing to tune. The grade is applied at compile time in one pass; a small change here affects every shot.
- **`LOOKBOOK_STYLE_KEYWORDS`** — short tokens prepended to every image/video prompt. Adding "shallow depth of field" or "motivated practical light" propagates everywhere.
- **`LOOKBOOK_PROMPT`** — what Claude is asked to produce as the visual bible. Editing this changes lens package, lighting style, reference films cited.
- **`SHOTLIST_SYSTEM`** / `shot_list_for_scene` — the system prompt that breaks scenes into shots. If `cinematography` is high, the issue is often shot variety: too many medium shots, no inserts, no coverage of reactions. The prompt enforces a hard 4/6/8 second duration choice.
- **Shot duration is capped at 8 seconds** — Veo 3.1's native single-call limit. The schema allows only `{4, 6, 8}`. Long beats are covered by multiple shots, not by extending one. `route_shot()` in `prepare.py` picks the Veo tier (Lite/Fast/Standard) based on `VEO_TIER`; you don't edit it.
- **`first_frame_prompt`** — the long prompt fed to GPT Image 2 + Nano Banana 2. If `continuity` is high (faces drifting), strengthen the identity-lock instruction.
- **`veo_prompt`** — the Veo prompt per shot/take. If `acting` or `sound` is high, dialogue blocks and performance hints live here.
- **`TAKE_VARIATIONS`** — per-take performance modifiers. Adding `"More urgency, faster blinks, shallower breath"` gives the editor different reads.
- **`MUSIC_STYLE`** — global music tone. If `sound` axis flags music, edit here.
- **Constants in `prepare.py`** — `TAKES_PER_SHOT`, `VEO_TIER`, `VEO_RESOLUTION`. These you CAN bump up via env vars when you want to throw more money at quality, but they're not in `produce.py` — set them in the shell before running.

## Discipline

- One or two changes per experiment, not five. You need to be able to attribute the delta in `film_loss` to a specific change.
- Keep a running log in `LOG.md` (you create and maintain this) of: experiment id, film_loss, what you changed, what the critic said, what you'll try next.
- If `film_loss` plateaus for 3 experiments in a row, change strategy or escalate to the human.
- If a stage in `produce.py` keeps failing for environmental reasons (API errors, timeout, etc.), don't iterate around it — flag it to the human.

## Workflow

```bash
# Manual single experiment + scoring (do this first to confirm the loop works)
python produce.py        # runs the pipeline, writes experiments/exp_NNN/, builds bible.pdf
python evaluate.py latest  # scores the film, refreshes bible.pdf with critique section

# Inspect the result
open experiments/exp_001/bible.pdf      # the canonical production document
cat experiments/exp_001/critique.md
cat experiments/exp_001/metric.json | jq '.scores, .film_loss, .changes'

# Then start iterating on produce.py and repeating.
```

The bible PDF is your debugging tool — flip through it and you'll see the look book, every shot's panel side-by-side with its rendered first frame, the editor's chosen take, and the critic's per-axis bar chart on the cover. Visual problems jump out faster from the bible than from a text critique.

## Notable patterns in the critic's output

- "Skin tones reading too cool" → tune `LOOKBOOK_GRADE` warm-midtone offset (`rm`/`gm`/`bm`).
- "Character X looks different in shots A/B/C" → strengthen `nano_banana` reference pass in `build_first_frames`, or add more `actor_photos/`.
- "Dialogue muddy under music" → reduce music volumex in `compile_final` from 0.20 to 0.12.
- "Pacing slack in scene N" → tighten `out_seconds` in EDL via `EDIT_SYSTEM` prompt asking for tighter cuts.
- "Shots too samey" → adjust `SHOTLIST_SYSTEM` to demand more size/angle variety.
- "Acting flat in close-ups" → expand `TAKE_VARIATIONS` with stronger emotional modifiers, bump `TAKES_PER_SHOT` env var.

## When to stop

You stop when:
1. The human says stop.
2. `film_loss` reaches < 0.20 (very good for AI-generated film, professional baseline).
3. Three consecutive experiments fail to improve `film_loss` and you've exhausted suggested changes.
4. Cost cap is hit.

Otherwise: read, edit, run, score, log, repeat.
