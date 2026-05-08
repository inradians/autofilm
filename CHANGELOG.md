# Changelog

## 0.2.0 — Runway-consolidated stack (May 2026)

The headline change: image, video, and SFX generation moved from four separate vendor APIs onto one Runway integration. Setup goes from 5 keys to 3, and the two annoying approval delays (OpenAI org verification, Google Cloud billing for Veo) are gone.

### Provider changes

| Capability | Before | After |
|---|---|---|
| Script / casting / lookbook / edit / stills critic | Anthropic `claude-opus-4-7` | Anthropic `claude-opus-4-7` *(unchanged)* |
| First-frame composition | OpenAI `gpt-image-2` | Runway `gpt_image_2` |
| Identity-lock / character refs | Google AI `gemini-3.1-flash-image-preview` (Nano Banana 2) | Runway `gemini_image3_pro` (Nano Banana) |
| Per-shot video | Google AI `veo-3.1-fast/standard-generate-preview` | Runway `veo3.1_fast` / `veo3.1` |
| Ambient SFX | ElevenLabs direct | Runway `eleven_text_to_sound_v2` |
| Music score | Stability `stable-audio-2.5` | Stability `stable-audio-2.5` *(unchanged — Runway has no music model)* |
| Long-video critic | Google AI `gemini-3-pro` | Google AI `gemini-3-pro` *(unchanged, but now optional)* |

### New capabilities

- **`gen4.5`** as a video tier — 12 credits/sec ($0.12/s), cheaper than Veo Fast, with **native reference-image identity-lock**. Set `VEO_TIER=gen4.5` or call `route_shot(.., tier="gen4.5")`. Trade-off: no native dialogue audio.
- **`seedance2`** as a video tier — 36 credits/sec ($0.36/s), supports **up to 15s in a single call**, lifting the previous 8s ceiling for the rare oner that needs it.
- **`gen4_aleph`** video-to-video transformation — exposed as `aleph_video_to_video()` in `prepare.py`. Run a per-shot regrade or stylistic transformation on top of any existing clip at 15 credits/sec ($0.15/s).
- **`gen4_image`** with native reference images — exposed as `runway_image(prompt, reference_images=[...], reference_tags=["jane"], model=GEN4_IMAGE_MODEL)`. Reference images are slot-tagged so prompts can address them as `@jane`. This is Runway's strongest identity-lock primitive and replaces the legacy `gpt_image → nano_banana` chain when continuity is the dominant failure mode.
- **`runway_tts()`** — narrator/voiceover generation via `eleven_multilingual_v2`. Useful for opening voiceover or letter-reading montages that Veo's in-frame audio can't cover.

### Behavioral changes

- Default for ambient SFX is now **off** (`AMBIENT_SFX_ENABLED=0`). Previously it was implicitly off too — gated on `ELEVENLABS_API_KEY` being set — but the new explicit toggle makes the intent clearer. Set `AMBIENT_SFX_ENABLED=1` to enable.
- `evaluate_film()` degrades gracefully when `GOOGLE_AI_API_KEY` is unset: the long-video Gemini critic is skipped and Claude-stills review becomes the sole reviewer. `film_loss` is still produced.
- `route_shot()` now accepts `tier="gen4.5"` and `tier="seedance2"`. Old tier names (`previs`, `fast`, `standard`) are unchanged.
- `_runway_uri()` (internal) auto-falls-back from base64 data URIs to ephemeral uploads when input bytes exceed Runway's per-type cap (5 MB image, 16 MB video, 32 MB audio).

### Backwards-compatible API surface

Everything `produce.py` imports still exists with the same name and signature, so existing custom `produce.py` files keep working without edits:

- `gpt_image(prompt, size, quality)` → now routes through Runway, `size` is mapped onto Runway pixel ratios.
- `nano_banana(prompt, reference_images)` → now uses `gemini_image3_pro` via Runway. Refs cap dropped from 14 to 3 (Runway's limit), but the new `gen4_image` path more than compensates.
- `veo(prompt, first_frame, ...)` → now routes through Runway.
- `elevenlabs_sfx(prompt, duration_seconds)` → now uses Runway's `eleven_text_to_sound_v2`.
- `claude_tool(...)`, `stable_audio(...)`, `route_shot(...)`, `plan_shot_durations(...)`, `extract_video_frame(...)`, `ffmpeg(...)`, `book_chunks(...)`, `veo_final_model(...)` — unchanged.

### Removed

- `openai>=1.50.0` and `elevenlabs>=1.10.0` from `pyproject.toml`. `runwayml>=3.0.0` added.
- `openai_client()` and `elevenlabs_client()` from `prepare.py`.
- `check_openai()` and `check_elevenlabs()` from `scripts/check_setup.py`. `check_runway()` added.

### Files added

- `.claude/skills/{rw-generate-video,rw-generate-image,rw-generate-audio,rw-setup-api-key,rw-api-reference}/` — vendored from `runwayml/skills` (MIT) for agent auto-discovery.
- `scripts/{generate_video,generate_image,generate_audio,get_task,list_models,runway_helpers}.py` — backing scripts the skills shell out to.
- `CHANGELOG.md` — this file.

### Cost impact

Default 3-scene experiment cost moves from ~$28 to ~$27. The shapes are slightly different but the totals are within rounding. Minimum-viable runs (`MAX_SCENES=1`, ambient off) stay at ~$6.

---

## 0.1.0 — initial release

Original autofilm pipeline on the multi-vendor April-2026 stack: Anthropic, OpenAI, Google AI, ElevenLabs, Stability.
