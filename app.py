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
            "<h1 style='margin-bottom:0;'>Suno Prompt Generator "
            "<span style='font-size:0.45em; font-weight:normal; color:#999;'>"
            "by AnimalMonk &nbsp; Discord.gg/StudioAI</span></h1>"
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
