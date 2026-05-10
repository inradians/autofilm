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

Two reviewers score each axis (Gemini 3 Pro on the actual video, Claude Opus 4.7 on stills) and the scores are averaged. CLIP character-drift is reported informationally but does not change `film_loss` directly. If `GOOGLE_AI_API_KEY` is unset, only Claude reviews — the metric is still produced, just from a single reviewer.

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

Each experiment with default `produce.py` costs roughly **$25-50** and takes **15-25 minutes** (mostly Runway video generation). At `MAX_SCENES=3` and `TAKES_PER_SHOT=1` you're at the cheap end.

Watch your spend. The loop is expensive enough that you should not run more than ~5 experiments per night without explicit human approval. All image/video/SFX is billed as Runway credits at $0.01 each — check the dashboard at https://dev.runwayml.com/ if you want a real-time view.

## The model menu

The Runway-consolidated stack gives you several knobs that didn't exist in the original pipeline. **Read this section before your first iteration** — half the high-leverage moves below depend on knowing which model to pick.

### Video models (set via `VEO_TIER` env var or call `route_shot(.., tier=...)` directly)

| Tier | Model | Cost/sec | Has dialogue audio? | Native ref-image support? | When to use |
|---|---|---|---|---|---|
| `previs` | `veo3.1_fast` | $0.15 | yes | no | cheap blocking validation (same as `fast`, kept for legacy) |
| `fast` *(default)* | `veo3.1_fast` | $0.15 | yes | no | every iteration |
| `standard` | `veo3.1` | $0.40 | yes | no | hero shot or final delivery |
| `gen4.5` | `gen4.5` | $0.12 | **no** | **yes** | identity-lock matters more than dialogue |
| `seedance2` | `seedance2` | $0.36 | no | yes | a beat that genuinely needs >8s in one cut |

**Picking between them based on critique:**

- "Faces drift across shots" / "character X looks different in shots A/B/C" → switch a few shots to `gen4.5` and pass the actor's reference image as an asset. Native multi-image identity lock is what Runway is famous for.
- "Lip-sync is off" / "dialogue feels disconnected" → stay on Veo Fast/Standard (only ones with native synced dialogue audio). Tighten the dialogue block in `veo_prompt()`.
- "This long beat would land better as a oner" → escalate that *one shot* to `seedance2` at 12-15s. Don't escalate the whole film — it's 2.4× the cost.
- "Acting flat" → bump `TAKES_PER_SHOT=3` (more options for the editor) before changing the model.
- "Color drifts between shots" → tune `LOOKBOOK_GRADE` first; if that's not enough, use `aleph_video_to_video()` as a per-shot regrade.

### Image models (used by `gpt_image()`, `nano_banana()`, `runway_image()`)

| Model id | Cost | Native references? | When to use |
|---|---|---|---|
| `gpt_image_2` | 1–41 c (high@1K=20) | optional, up to 3 | first-frame composition — strongest instruction-following |
| `gemini_image3_pro` | 20 c @ 1K/2K | optional, up to 3 | identity lock, multi-image fusion |
| `gen4_image` | 5 c @ 720p, 8 c @ 1080p | optional, up to 3 | Runway's flagship — strong character continuity |
| `gen4_image_turbo` | 2 c | **required** | cheap iteration when you already have refs |
| `gemini_2.5_flash` | 5 c | optional, up to 3 | Nano Banana, fastest |

**`gen4_image` with reference images is the single biggest continuity unlock.** The original pipeline emulated this by chaining `gpt_image` → `nano_banana(refs)`. The new direct path is one call: `runway_image(prompt, reference_images=[actor_png], reference_tags=["jane"], model=GEN4_IMAGE_MODEL)`. Then in the prompt you can address that ref as `@jane`. If continuity is the dominant axis flagged by the critic, route `build_first_frames()` through `gen4_image` instead of the legacy `gpt_image → nano_banana` chain.

### Video-to-video transformation (NEW)

`aleph_video_to_video(prompt, input_video_bytes, reference_image=None)` runs Gen-4 Aleph at 15 c/s ($0.15/s) on top of an existing clip. Use cases the old pipeline couldn't do:

- **Per-shot regrade** when the global ffmpeg `LOOKBOOK_GRADE` chain doesn't hit a specific shot well. Ask for "warm golden hour, soft contrast" and Aleph rewrites the lighting.
- **Seasonal/temporal transformation** — same blocking, different time of day or weather.
- **Stylistic recovery** — when one shot ended up looking like a different film than the rest, regrade just that one.

Don't chain Aleph reflexively — it doubles the cost of the affected shot and adds 1-2 minutes of wall-clock. Use it when the critic explicitly flags a per-shot color or lighting failure.

### Audio additions

- `elevenlabs_sfx(prompt, duration_seconds)` — ambient bed; off by default, enable with `AMBIENT_SFX_ENABLED=1`.
- `runway_tts(text, voice_id)` — **NEW** narrator/voiceover. Veo's native audio only covers in-frame dialogue. If you want an opening voiceover or letter-reading montage, this is the tool.

### Transitions (new in v0.2)

Each shot in the storyboard can carry a `transition_out` field that defines how it bridges to the next shot. Default is `cut` (hard cut). The catalog lives in `transitions.py` and is auto-injected into the SHOTLIST_SYSTEM prompt, so Claude has the full menu when planning.

Editorial vocabulary, in rough frequency order:

| Use it for | Transition |
|---|---|
| 95% of shot-to-shot bridges | `cut` |
| End of a chapter / major time jump / death | `fadeblack` (0.7-1.5s) |
| Flashback entry / dream sequence / revelation | `fadewhite` (0.5-1.0s) |
| Continuity within a montage | `fade` or `dissolve` (0.3-0.6s) |
| Memory beat | `fadegrays` (0.6-1.0s) |
| "Meanwhile" geographic move | `wiperight` or `slideleft` (0.4-0.7s) |
| Sequence button / iris close | `circleclose` (0.5-1.0s) |
| Drug/sleep/glitch | `hblur` or `pixelize` (0.3-0.6s) |

**When the critic flags pacing or structure**, transitions are often the right knob:

- "Scene transitions feel abrupt" → add `fadeblack` (0.7s) on the last shot of each scene.
- "Montage doesn't read as a montage" → switch internal cuts to `fade` (0.3s) for connective tissue.
- "Time jump unclear in scene N" → put a `fadeblack` or `fadegrays` at the moment the leap happens.
- "This sequence drags" → don't add transitions, *remove* them. Hard cuts are tighter than dissolves.

**Anti-pattern**: a film with a transition on every shot reads as amateurish. Reserve them for scene endings and the rare beat that genuinely needs one. The default of `cut` is correct for ~95% of shots.

The compile step auto-detects whether any scene contains non-cut transitions. If yes, that scene is rendered through ffmpeg's `xfade`/`acrossfade` filter chain (one re-encode pass). If no, the scene takes the fast moviepy concat path. So transitions cost wall-clock only on scenes that use them.

**Custom GLSL transitions** (for the rare beat that ffmpeg can't do): two ship out of the box — `chromatic_glitch` (RGB-split, signal-loss/digital-intrusion beats) and `displacement_push` (luminance-driven dissolution, memory or POV shifts). Use them like any other transition (`{"type": "chromatic_glitch", "duration": 0.8}`). When the storyboard names a GLSL transition the renderer switches to a slower pairwise path (~1s per junction at 720p on a CPU; effectively instant on a GPU). New shaders can be registered from `produce.py` by calling `register_glsl_transition(name, shader, ...)` with a gl-transitions.com style fragment shader — the registration goes into the experiment snapshot so it's reproducible. Keep this rare: GLSL transitions are a creative knob, not a default.

## Where to make changes

The critic's `changes` list names targets specific to `produce.py`. Common high-leverage knobs:

- **`DIRECTOR` / `CINEMATOGRAPHER`** — env-var creative-direction inputs. When set, the look book stage derives concrete craft markers (lens choice, lighting approach, palette, framing patterns) from the named artists' bodies of work and bakes those into `lookbook.json`. Use this when the critic says the visual identity is generic. Don't toggle these mid-iteration loop unless you mean to start fresh — they shift everything downstream.
- **`LOOKBOOK_GRADE`** — ffmpeg color filter chain. If `color` axis is high, this is the first thing to tune. The grade is applied at compile time in one pass; a small change here affects every shot.
- **`LOOKBOOK_STYLE_KEYWORDS`** — short tokens prepended to every image/video prompt. Adding "shallow depth of field" or "motivated practical light" propagates everywhere.
- **`LOOKBOOK_PROMPT`** — what Claude is asked to produce as the visual bible. Editing this changes lens package, lighting style, reference films cited.
- **`SHOTLIST_SYSTEM`** / `shot_list_for_scene` — the system prompt that breaks scenes into shots. If `cinematography` is high, the issue is often shot variety: too many medium shots, no inserts, no coverage of reactions. The prompt enforces a hard 4/6/8 second duration choice.
- **Shot duration is capped at 8 seconds** unless you explicitly route via `seedance2`. The schema for Veo allows only `{4, 6, 8}`. Long beats are covered by multiple shots, not by extending one. `route_shot()` in `prepare.py` picks the tier based on `VEO_TIER`; you don't edit it, you just pass `tier=...` per shot when you want to deviate from the global default.
- **`first_frame_prompt`** — the long prompt fed to the image generator. If `continuity` is high (faces drifting), this is your primary lever. Two moves:
  1. Strengthen the identity-lock instruction in the prompt itself.
  2. Switch the image call from `gpt_image` → `runway_image(.., reference_images=[ref], model=GEN4_IMAGE_MODEL)` so refs go in via Runway's native referenceImages slot rather than the legacy nano-banana chain.
- **`veo_prompt`** — the prompt per shot/take. If `acting` or `sound` is high, dialogue blocks and performance hints live here.
- **`TAKE_VARIATIONS`** — per-take performance modifiers. Adding `"More urgency, faster blinks, shallower breath"` gives the editor different reads.
- **`MUSIC_STYLE`** — global music tone. If `sound` axis flags music, edit here.
- **Constants in `prepare.py`** — `TAKES_PER_SHOT`, `VEO_TIER`, `VEO_RESOLUTION`. These you CAN bump up via env vars when you want to throw more money at quality, but they're not in `produce.py` — set them in the shell before running.

## Discipline

- One or two changes per experiment, not five. You need to be able to attribute the delta in `film_loss` to a specific change.
- Keep a running log in `LOG.md` (already scaffolded — append to it) of: experiment id, film_loss, what you changed, what the critic said, what you'll try next.
- If `film_loss` plateaus for 3 experiments in a row, change strategy or escalate to the human.
- If a stage in `produce.py` keeps failing for environmental reasons (API errors, timeout, etc.), don't iterate around it — flag it to the human. The pipeline is designed to crash loudly on infrastructure problems.
- **Don't change the video model on every run.** Switching `VEO_TIER=fast` ↔ `gen4.5` is a structural change — every shot rerolls. Use it deliberately when the critic specifically flags continuity, not as a vibes-based toggle.

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
- "Character X looks different in shots A/B/C" → switch first-frame generation to `gen4_image` with a stable reference image for X. Or strengthen the existing nano-banana lock.
- "Dialogue muddy under music" → reduce music volumex in `compile_final` from 0.20 to 0.12.
- "Pacing slack in scene N" → tighten `out_seconds` in EDL via `EDIT_SYSTEM` prompt asking for tighter cuts.
- "Shots too samey" → adjust `SHOTLIST_SYSTEM` to demand more size/angle variety.
- "Acting flat in close-ups" → expand `TAKE_VARIATIONS` with stronger emotional modifiers, bump `TAKES_PER_SHOT` env var.
- "Lighting wrong on shot M" (single shot) → wrap that shot in `aleph_video_to_video()` as a regrade pass before EDL assembly.
- "This shot needs to breathe" (single oner) → escalate that one shot's tier to `seedance2` for a 12-15s single take.
- "Scene boundaries feel abrupt" / "no breath between scenes" → set `transition_out: {type: "fadeblack", duration: 0.7}` on the last shot of each scene in the storyboard.
- "Time jump in scene N is confusing" → put `fadeblack` or `fadegrays` on the shot right before the jump.
- "Montage doesn't read as a montage" → set `transition_out: {type: "fade", duration: 0.3}` on every internal montage shot.

## When to stop

You stop when:
1. The human says stop.
2. `film_loss` reaches < 0.20 (very good for AI-generated film, professional baseline).
3. Three consecutive experiments fail to improve `film_loss` and you've exhausted suggested changes.
4. Cost cap is hit.

Otherwise: read, edit, run, score, log, repeat.
