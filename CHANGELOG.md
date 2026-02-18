# Changelog

## [1.0.0] - 2025-02-18

### Added
- Initial release of Suno Prompt Generator
- Gradio web UI with dark theme and orange accent
- OpenRouter API integration using OpenAI-compatible SDK
- Model selector with 5 presets:
  - Google Gemini 3 Pro
  - Google Gemini 3 Flash
  - Anthropic Claude Sonnet 4.6
  - OpenAI GPT-5.2
  - xAI Grok 4
  - Custom model ID input
- Natural language song idea input
- Weirdness slider (0-100) controlling creative hallucination intensity:
  - 0-20: Conventional, straightforward prompts
  - 21-50: Mild — unusual genre combos, abstract descriptors
  - 51-80: Medium — genre collisions, paradox emotions, impossible textures, surreal environments
  - 81-100: Maximum — full hallucination arsenal with abstract directives
- Five output sections, each with copy button:
  - **Song Title** — AI-generated evocative title
  - **Style Prompt** — paste directly into Suno's Style Prompt field
  - **Lyrics with Tags** — paste directly into Suno's Lyrics field, includes structure tags and performance cues
  - **Suno UI Settings** — recommended Weirdness and Style Influence values (0-100) with reasoning
  - **Cover Art Image Prompt** — rich visual description for AI image generation (Grok, Midjourney, DALL-E)
- Consolidated knowledge base from three source documents:
  - Suno Database (genre/subgenre trees, instruments, vocals, production parameters)
  - Suno AI Prompt Mechanics Guide (tag weighting, character limits, working vs ignored terms)
  - Suno Prompt Generator / Gemini Instructions (creative hallucination framework)
- Knowledge base covers: genres, subgenres, genre qualities, genre fusion strategy, era modifiers, working production terms (drums, bass, reverb, delay, texture, synths, guitars, keys, strings, percussion, compression, tonal balance), performance & humanization (microtiming, velocity, energy, arrangement density/layering/focus), vocal treatment (delivery, registers, harmony, articulation, effects, persona tags), structure tags (sections, bar counts, performance cues, energy/mood/harmony/lyric tone/chord tags), harmony/tempo/groove (key, modes, BPM, groove feel, drum style, time signatures), language/accent/narrative (accent, diction, POV, localization), prompt construction priority order, and common fixes
- `.env` file for API key storage
- `Launch Suno Prompter.bat` for double-click startup
