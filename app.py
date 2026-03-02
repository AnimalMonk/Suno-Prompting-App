"""
Suno Prompting App
Converts natural language song ideas into structured Suno AI prompts.
Supports OpenRouter and ElectronHub APIs with auto-detection.
"""

import json
import os
import random
import re
import tempfile

import gradio as gr
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from knowledge_base import build_system_prompt

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

MODELS = {
    "Google Gemini 3 Pro": "google/gemini-3-pro-preview",
    "Google Gemini 3 Flash": "google/gemini-3-flash-preview",
    "Anthropic Claude Sonnet 4.6": "anthropic/claude-sonnet-4.6",
    "OpenAI GPT-5.2": "openai/gpt-5.2",
    "xAI Grok 4": "x-ai/grok-4",
    "Custom": "custom",
}

FREE_MODELS = [
    "glm-4.5-air",
    "qwen3-coder-480b-a35b-instruct:free",
    "llama-4-maverick-17b-128e-instruct",
    "claude-3-haiku-20240307",
    "kimi-k2.5:free",
    "gemini-2.5-flash",
]

# ElectronHub uses different model IDs for some models.
# Most just drop the provider prefix, but these two need explicit overrides.
ELECTRONHUB_MODEL_OVERRIDES = {
    "anthropic/claude-sonnet-4.6": "claude-sonnet-4-6",
    "x-ai/grok-4": "grok-4-0709",
}

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.orange,
    secondary_hue=gr.themes.colors.neutral,
    neutral_hue=gr.themes.colors.gray,
    font=gr.themes.GoogleFont("Inter"),
).set(
    body_background_fill="#1a1a1a",
    body_background_fill_dark="#1a1a1a",
    body_text_color="#e0e0e0",
    body_text_color_dark="#e0e0e0",
    block_background_fill="#2a2a2a",
    block_background_fill_dark="#2a2a2a",
    block_border_color="#444",
    block_border_color_dark="#444",
    block_label_text_color="#ccc",
    block_label_text_color_dark="#ccc",
    block_title_text_color="#fff",
    block_title_text_color_dark="#fff",
    input_background_fill="#333",
    input_background_fill_dark="#333",
    input_border_color="#555",
    input_border_color_dark="#555",
    button_primary_background_fill="#e67e22",
    button_primary_background_fill_dark="#e67e22",
    button_primary_background_fill_hover="#d35400",
    button_primary_background_fill_hover_dark="#d35400",
    button_primary_text_color="#fff",
    button_primary_text_color_dark="#fff",
)


# ─────────────────────────────────────────────
# CORE LOGIC
# ─────────────────────────────────────────────

def _get_client(api_key: str = "", free_mode: bool = False):
    """Create OpenAI-compatible client. Auto-detects provider from key prefix.

    Free mode uses the hardcoded ElectronHub key.
    Otherwise: ek-* -> ElectronHub, everything else -> OpenRouter.
    """
    if free_mode:
        key = os.getenv("ELECTRONHUB_API_KEY", "")
        if not key:
            raise ValueError(
                "Free Mode is unavailable. Enter your own API key and uncheck Free Mode."
            )
        base_url = "https://api.electronhub.ai/v1"
    else:
        key = api_key.strip()
        if not key:
            raise ValueError("No API key provided. Enter your OpenRouter or ElectronHub API key above.")
        # Auto-detect provider from key prefix
        if key.startswith("ek-"):
            base_url = "https://api.electronhub.ai/v1"
        else:
            base_url = "https://openrouter.ai/api/v1"

    return OpenAI(base_url=base_url, api_key=key, timeout=90.0)


def _resolve_model_id(model_id: str) -> str:
    """Translate OpenRouter model ID to ElectronHub format.

    Most models just drop the provider prefix (google/, anthropic/, etc.).
    A few need explicit overrides (different naming conventions).
    """
    if model_id in ELECTRONHUB_MODEL_OVERRIDES:
        return ELECTRONHUB_MODEL_OVERRIDES[model_id]
    if "/" in model_id:
        return model_id.split("/", 1)[1]
    return model_id


def _fix_json_newlines(text: str) -> str:
    """Replace literal newlines inside JSON string values with \\n.

    Free models often put real line breaks in string fields (especially lyrics)
    instead of \\n escape sequences, which makes json.loads() fail.
    """
    result = []
    in_string = False
    escape_next = False
    for char in text:
        if escape_next:
            result.append(char)
            escape_next = False
            continue
        if char == '\\':
            result.append(char)
            escape_next = True
            continue
        if char == '"':
            in_string = not in_string
            result.append(char)
            continue
        if char == '\n' and in_string:
            result.append('\\n')
            continue
        result.append(char)
    return ''.join(result)


def _extract_json(raw: str) -> dict:
    """Extract JSON object from model response, even if wrapped in extra text.

    Handles: clean JSON, markdown code fences, preamble/trailing text,
    and literal newlines inside string values (common with free models).
    """
    # Strip markdown code fences
    cleaned = re.sub(r'^```(?:json)?\s*\n?', '', raw.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r'\n?```\s*$', '', cleaned.strip(), flags=re.MULTILINE)

    # Try parsing cleaned text directly
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Fix literal newlines inside string values, then try again
    try:
        return json.loads(_fix_json_newlines(cleaned))
    except json.JSONDecodeError:
        pass

    # Find the first { ... } block (handles preamble/trailing text)
    match = re.search(r'\{[\s\S]*\}', cleaned)
    if match:
        block = match.group()
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            pass
        try:
            return json.loads(_fix_json_newlines(block))
        except json.JSONDecodeError:
            pass

    raise json.JSONDecodeError("No valid JSON found in response", raw, 0)


def _generate_cover_image(prompt: str):
    """Generate cover art via ElectronHub SDXL. Returns file path or None on failure."""
    if not prompt:
        return None

    eh_key = os.getenv("ELECTRONHUB_API_KEY", "")
    if not eh_key:
        return None  # Silently skip — image gen is a bonus, not critical

    try:
        resp = requests.post(
            "https://api.electronhub.ai/v1/images/generations",
            headers={
                "Authorization": f"Bearer {eh_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "sdxl",
                "prompt": prompt,
                "n": 1,
                "size": "1024x1024",
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        image_url = data["data"][0]["url"]

        # Download image to temp file for Gradio
        img_resp = requests.get(image_url, timeout=30)
        img_resp.raise_for_status()

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.write(img_resp.content)
        tmp.close()
        return tmp.name

    except Exception:
        return None  # Image gen failure should never block the main output


def generate_prompt(
    api_key: str,
    song_idea: str,
    model_choice: str,
    custom_model: str,
    weirdness: int,
    free_mode: bool,
):
    """Call LLM API, parse structured response, and generate cover art image."""
    # 6 outputs: title, style, lyrics, settings, cover_art_image, cover_art_text
    if not song_idea.strip():
        return "", "Please enter a song idea.", "", "", None, ""

    # Resolve model ID
    if free_mode:
        model_id = None  # Set during cascade
    elif model_choice == "Custom":
        model_id = custom_model.strip()
        if not model_id:
            return "", "Please enter a custom model ID.", "", "", None, ""
    else:
        model_id = MODELS.get(model_choice, "google/gemini-3-flash-preview")

    system_prompt = build_system_prompt(weirdness)

    try:
        client = _get_client(api_key, free_mode=free_mode)
    except ValueError as e:
        return "", str(e), "", "", None, ""

    # Translate model ID for ElectronHub (different naming convention)
    if not free_mode and model_id:
        key = api_key.strip()
        if key.startswith("ek-"):
            model_id = _resolve_model_id(model_id)

    # --- LLM call ---
    raw = None
    if free_mode:
        # Shuffle and cascade through free models until one works
        models_to_try = FREE_MODELS[:]
        random.shuffle(models_to_try)
        last_error = None
        for model_id in models_to_try:
            try:
                response = client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": song_idea},
                    ],
                    temperature=0.9,
                    max_tokens=4096,
                    timeout=90.0,
                )
                raw = response.choices[0].message.content.strip()
                print(f"Free Mode: {model_id} succeeded")
                break
            except Exception as e:
                last_error = e
                print(f"Free Mode: {model_id} failed ({e}), trying next...")
                continue
        if raw is None:
            return "", f"All free models failed. Last error: {last_error}", "", "", None, ""
    else:
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": song_idea},
                ],
                temperature=0.9,
                max_tokens=4096,
                timeout=90.0,
            )
            raw = response.choices[0].message.content.strip()
        except Exception as e:
            return "", f"Error: {e}", "", "", None, ""

    # --- Parse JSON response ---
    try:
        data = _extract_json(raw)

        song_title = data.get("song_title", "Untitled")
        style_prompt = data.get("style_prompt", "")
        lyrics = data.get("lyrics", "")

        w = data.get("weirdness", "N/A")
        w_reason = data.get("weirdness_reasoning", "")
        si = data.get("style_influence", "N/A")
        si_reason = data.get("style_influence_reasoning", "")
        settings = f"Weirdness: {w}/100\n{w_reason}\n\nStyle Influence: {si}/100\n{si_reason}"

        cover_art_text = data.get("cover_art_prompt", "")

        # Generate cover art image (failure is silent, never blocks text outputs)
        cover_art_image = _generate_cover_image(cover_art_text)

        return song_title, style_prompt, lyrics, settings, cover_art_image, cover_art_text

    except json.JSONDecodeError:
        return (
            "",
            f"[JSON parse error - raw response below]\n\n{raw}",
            "",
            "",
            None,
            "",
        )
    except Exception as e:
        return "", f"Error: {e}", "", "", None, ""


def toggle_custom_visibility(choice):
    """Show/hide custom model text field."""
    return gr.update(visible=(choice == "Custom"))


def toggle_free_mode(free_mode: bool):
    """When Free Mode is checked, gray out API key and model dropdown."""
    return (
        gr.update(interactive=not free_mode),  # api_key_input
        gr.update(interactive=not free_mode),  # model_dropdown
    )


# ─────────────────────────────────────────────
# APP BUILDER
# ─────────────────────────────────────────────

def create_app():
    """Build and return the Gradio Blocks app and theme."""
    with gr.Blocks(title="Suno Prompt Generator") as demo:
        gr.HTML(
            '<div style="display:flex;align-items:center;gap:16px;">'
            '<img src="data:image/png;base64,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" alt="s.AI logo" style="height:80px;border-radius:12px;">'
            '<div>'
            '<h1 style="margin:0;">Suno Prompt Generator</h1>'
            '<p style="margin:4px 0 0 0;font-size:0.9em;">by AnimalMonk &nbsp;|&nbsp; '
            '<a href="https://discord.gg/StudioAI" target="_blank">Join us on Discord</a></p>'
            '</div></div>'
        )
        gr.Markdown(
            "This is meant to spark new ideas or get you a starting point. Take what it gives you and make it goldensome!\n\n"
            "Works with [OpenRouter](https://openrouter.ai/keys) and "
            "[ElectronHub](https://api.electronhub.ai) API keys. "
            "Your key is sent directly to the provider and is never stored. "
            "Or check **Free Mode** below to try it without any key!"
        )

        with gr.Accordion("API Key", open=True):
            api_key_input = gr.Textbox(
                label="API Key (OpenRouter or ElectronHub)",
                placeholder="sk-or-v1-... or ek-...",
                type="password",
                lines=1,
            )

        free_mode_checkbox = gr.Checkbox(
            label="Free Mode",
            value=False,
        )

        with gr.Row():
            model_dropdown = gr.Dropdown(
                choices=list(MODELS.keys()),
                value="Google Gemini 3 Flash",
                label="AI Model",
                scale=2,
            )
            custom_model_input = gr.Textbox(
                label="Custom Model ID",
                placeholder="e.g. meta-llama/llama-4-maverick",
                visible=False,
                scale=2,
            )

        song_input = gr.Textbox(
            label="Song Idea: describe your song idea in natural language. Get back structured Suno prompts.",
            placeholder="A melancholy song about driving alone at night on empty highways, with a female vocal that sounds tired but hopeful...",
            lines=4,
        )

        weirdness_slider = gr.Slider(
            minimum=0,
            maximum=100,
            value=30,
            step=1,
            label="Weirdness (0 = conventional, 100 = maximum creative hallucination)",
        )

        generate_btn = gr.Button("Generate Suno Prompt", variant="primary", size="lg")

        gr.Markdown("---")

        style_output = gr.Textbox(
            label="Style Prompt (paste into Suno's Style Prompt field)",
            lines=5,
            buttons=["copy"],
            interactive=False,
        )

        title_output = gr.Textbox(
            label="Song Title",
            lines=1,
            buttons=["copy"],
            interactive=False,
        )

        lyrics_output = gr.Textbox(
            label="Lyrics with Tags (paste into Suno's Lyrics field)",
            lines=20,
            buttons=["copy"],
            interactive=False,
        )

        with gr.Row():
            settings_output = gr.Textbox(
                label="Suno UI Settings",
                lines=5,
                interactive=False,
                scale=1,
            )

        gr.Markdown("---")

        gr.Markdown("### Cover Art")
        with gr.Row():
            with gr.Column(scale=1):
                cover_art_image = gr.Image(
                    label="Generated Cover Art",
                    type="filepath",
                    interactive=False,
                    height=512,
                )
            with gr.Column(scale=1):
                cover_art_output = gr.Textbox(
                    label="Cover Art Image Prompt (paste into Grok or image generator)",
                    lines=6,
                    buttons=["copy"],
                    interactive=False,
                )

        # Events
        model_dropdown.change(
            fn=toggle_custom_visibility,
            inputs=model_dropdown,
            outputs=custom_model_input,
        )

        free_mode_checkbox.change(
            fn=toggle_free_mode,
            inputs=free_mode_checkbox,
            outputs=[api_key_input, model_dropdown],
        )

        outputs = [title_output, style_output, lyrics_output, settings_output, cover_art_image, cover_art_output]
        inputs = [api_key_input, song_input, model_dropdown, custom_model_input, weirdness_slider, free_mode_checkbox]

        generate_btn.click(
            fn=generate_prompt,
            inputs=inputs,
            outputs=outputs,
        )

        song_input.submit(
            fn=generate_prompt,
            inputs=inputs,
            outputs=outputs,
        )

    return demo, THEME


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

# Build the app at module level so HF Spaces can detect it
demo, _theme = create_app()

if __name__ == "__main__":
    import sys
    import time
    import threading
    import webbrowser

    _server_url = "http://127.0.0.1:7860"
    _shutdown = threading.Event()

    # ── System tray icon ──
    def _start_tray():
        try:
            import pystray
            from PIL import Image, ImageDraw

            # Create a small orange icon with "S" on it
            img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            draw.rounded_rectangle([4, 4, 60, 60], radius=12, fill="#FF8C00")
            draw.text((20, 12), "S", fill="white")

            def on_open(icon, item):
                webbrowser.open(_server_url)

            def on_quit(icon, item):
                print("🛑 Quit from system tray.")
                _shutdown.set()
                icon.stop()

            icon = pystray.Icon(
                "suno_prompt_gen",
                img,
                "Suno Prompt Generator",
                menu=pystray.Menu(
                    pystray.MenuItem("Open in Browser", on_open, default=True),
                    pystray.MenuItem("Quit", on_quit),
                ),
            )
            icon.run()
        except Exception as e:
            print(f"⚠️ System tray unavailable: {e}")

    tray_thread = threading.Thread(target=_start_tray, daemon=True)
    tray_thread.start()

    # ── Server loop ──
    first_launch = True
    while not _shutdown.is_set():
        try:
            print(f"{'🚀 Launching' if first_launch else '🔄 Relaunching'} Suno Prompt Generator...")
            demo, _theme = create_app()
            demo.launch(inbrowser=first_launch, theme=_theme, quiet=False)
            # If launch() returns, the server stopped
            if not _shutdown.is_set():
                print("⚠️ Gradio server stopped. Restarting in 3s...")
        except Exception as e:
            if not _shutdown.is_set():
                print(f"❌ Gradio crashed: {e}. Restarting in 3s...")
        first_launch = False
        if not _shutdown.is_set():
            time.sleep(3)

    print("👋 Suno Prompt Generator shut down.")
    sys.exit(0)
