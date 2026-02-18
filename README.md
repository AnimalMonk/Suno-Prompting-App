---
title: Suno Prompt Generator
emoji: 🎵
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: "6.6.0"
app_file: app.py
pinned: false
license: mit
---

# Suno Prompt Generator

Describe your song idea in natural language. Get back structured prompts optimized for [Suno AI](https://suno.com) music generation.

## Features

- **Style Prompt** — paste directly into Suno's Style Prompt field
- **Song Title** — AI-generated evocative title
- **Lyrics with Tags** — complete lyrics with structure tags, performance cues, and humanization
- **Suno UI Settings** — recommended Weirdness and Style Influence values with reasoning
- **Cover Art Prompt** — rich visual description for AI image generation (Grok, Midjourney, DALL-E)
- **Weirdness Slider** (0-100) — from conventional to maximum creative hallucination
- **Model Selection** — choose from Gemini, Claude, GPT, Grok, or any custom OpenRouter model

## Usage

1. Get an [OpenRouter API key](https://openrouter.ai/keys)
2. Paste your key into the API Key field
3. Describe your song idea
4. Adjust the Weirdness slider
5. Click Generate

Your API key is sent directly to OpenRouter and is never stored.

## Run Locally

```bash
git clone https://github.com/AnimalMonk/Suno-Prompting-App
cd Suno-Prompting-App
python -m venv venv
venv\Scripts\pip install -r requirements.txt
# Create .env with: OPENROUTER_API_KEY=your-key-here
venv\Scripts\python app.py
```

Or double-click `Launch Suno Prompter.bat` on Windows.

## Windows Installer

Download the installer from [Releases](https://github.com/AnimalMonk/Suno-Prompting-App/releases). It prompts for your API key during setup — no Python needed.
