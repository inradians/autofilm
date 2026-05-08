# .claude/skills — vendored Runway agent skills

This directory ships a subset of [`runwayml/skills`](https://github.com/runwayml/skills) (MIT-licensed, see `scripts/RUNWAY_SCRIPTS_LICENSE` at the project root) so an agent working in this repo (Claude Code, Cursor, etc.) auto-discovers them.

## What's here

| Skill | Purpose |
|---|---|
| `rw-generate-video/` | One-off video generations via `gen4.5`, `seedance2`, `veo3.1*`, `gen4_aleph`. |
| `rw-generate-image/` | One-off image generations via `gen4_image`, `gemini_image3_pro`, `gpt_image_2`, `gemini_2.5_flash`. |
| `rw-generate-audio/` | TTS, SFX, voice isolation, dubbing, voice conversion. |
| `rw-setup-api-key/` | Walkthrough for getting the API key. |
| `rw-api-reference/` | Reference for raw HTTP calls (when the SDK isn't enough). |

The skills shell out to runnable Python scripts at the project root: `scripts/generate_video.py`, `scripts/generate_image.py`, `scripts/generate_audio.py`, `scripts/runway_helpers.py`, `scripts/get_task.py`, `scripts/list_models.py`. These were vendored from `runwayml/skills/scripts/` and are MIT-licensed (see `scripts/RUNWAY_SCRIPTS_LICENSE`).

## How the agent uses these vs. `prepare.py`

The skills are for **one-off, exploratory generations** outside the main pipeline. For example, when iterating on a single problematic shot, the agent might want to test a new prompt against `gen4.5` directly without rerunning the whole pipeline. That's a job for `rw-generate-video`:

```bash
uv run scripts/generate_video.py \
  --prompt "warm golden hour, slow dolly in" \
  --image-url ./experiments/exp_003/frames/scene_01/shot_03.png \
  --filename test.mp4 \
  --model gen4.5 \
  --duration 5
```

The actual production pipeline goes through `prepare.py`'s helpers (`veo()`, `runway_image()`, `aleph_video_to_video()`, etc.), which use the official `runwayml` Python SDK and are integrated with the artifact-cache + prompt-log machinery.

**Rule of thumb:**

- Editing `produce.py` for the next experiment → use `prepare.py` helpers (they log prompts to `prompts.json`, write bytes to the experiment dir, integrate with the bible).
- Spitballing a one-off prompt to see what it looks like → use `scripts/generate_*.py` directly.

Both paths read the same `RUNWAYML_API_SECRET` from the environment.

## Updating

These were vendored on the day the project was migrated to Runway. To pull in upstream changes:

```bash
git clone --depth 1 https://github.com/runwayml/skills.git /tmp/runway-skills
cp -r /tmp/runway-skills/skills/{rw-generate-video,rw-generate-image,rw-generate-audio,rw-setup-api-key,rw-api-reference} .claude/skills/
cp /tmp/runway-skills/scripts/*.py scripts/
```

Make sure to also re-check `prepare.py` for any new model IDs that should be added to the `VIDEO_MODELS` registry or the bible's display-name map.

