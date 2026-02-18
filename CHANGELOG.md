# Changelog

## [1.2.0] - 2026-02-18

### Added
- **Hugging Face Spaces support** — app can now be deployed as a public web app
- API key input field in the UI (users bring their own OpenRouter key)
- `README.md` with HF Spaces YAML frontmatter and project documentation
- Privacy note: API keys are sent directly to OpenRouter and never stored

### Changed
- `generate_prompt()` now accepts `api_key` parameter from the UI instead of only reading from env
- `_get_client()` prioritizes UI-provided key, falls back to `.env` for local/desktop use
- Gradio `demo` object created at module level for HF Spaces auto-detection
- API Key section uses collapsible `gr.Accordion` and password-masked input

## [1.1.2] - 2026-02-18

### Fixed
- Fixed second installer crash: `groovy/version.txt` also missing from PyInstaller bundle
- Added `--collect-data groovy` to build script
- Audited all dependencies for missing data files to prevent further issues

## [1.1.1] - 2026-02-18

### Fixed
- Fixed installer crash on launch: `safehttpx/version.txt` was not being bundled by PyInstaller, causing a `FileNotFoundError` on the target machine
- Added `--collect-data safehttpx` to build script

## [1.1.0] - 2026-02-18

### Added
- **Windows installer packaging** via PyInstaller + Inno Setup
  - `desktop_app.py` — Packaged entry point that launches Gradio in default browser with no console window
  - `build.bat` — One-click build script that runs PyInstaller and optionally Inno Setup
  - `installer.iss` — Inno Setup script with custom API key prompt during installation
  - `hooks/hook-gradio.py` — PyInstaller hook for Gradio source collection
  - `runtime_hook.py` — Multiprocessing freeze support for PyInstaller
- Installer features:
  - Professional setup wizard with custom API key input page
  - Start Menu and optional desktop shortcuts
  - Add/Remove Programs entry
  - Writes `.env` file with API key to install directory
  - Option to launch app after installation
- Build produces both a portable exe (`dist/SunoPromptGenerator/`) and an installer (`Output/SunoPromptGenerator_Setup.exe`)

### Changed
- Refactored `app.py` to expose `create_app()` function, allowing reuse by both dev mode and desktop packaging
- Updated `.gitignore` to exclude build artifacts (`build/`, `dist/`, `Output/`, `*.spec`) and source reference documents
- Removed `pywebview` from requirements (pythonnet incompatible with Python 3.14; using browser-based approach instead)

## [1.0.1] - 2025-02-18

### Changed
- Added CLAUDE.md to .gitignore to keep project instructions local only

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
