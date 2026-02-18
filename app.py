"""
Suno Prompting App
Converts natural language song ideas into structured Suno AI prompts via OpenRouter API.
"""

import json
import os

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

from knowledge_base import build_system_prompt

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

MODELS = {
    "Google Gemini 3 Pro": "google/gemini-3-pro-preview",
    "Google Gemini 3 Flash": "google/gemini-3-flash-preview",
    "Anthropic Claude Sonnet 4.6": "anthropic/claude-sonnet-4.6",
    "OpenAI GPT-5.2": "openai/gpt-5.2",
    "xAI Grok 4": "x-ai/grok-4",
    "Custom": "custom",
}

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)


# ─────────────────────────────────────────────
# CORE LOGIC
# ─────────────────────────────────────────────

def generate_prompt(song_idea: str, model_choice: str, custom_model: str, weirdness: int):
    """Call OpenRouter and parse the structured response."""
    if not song_idea.strip():
        return "", "Please enter a song idea.", "", "", ""

    # Resolve model ID
    if model_choice == "Custom":
        model_id = custom_model.strip()
        if not model_id:
            return "", "Please enter a custom model ID.", "", "", ""
    else:
        model_id = MODELS.get(model_choice, "google/gemini-3-flash-preview")

    system_prompt = build_system_prompt(weirdness)

    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": song_idea},
            ],
            temperature=0.9,
            max_tokens=4096,
        )

        raw = response.choices[0].message.content.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            lines = raw.split("\n")
            # Remove first line (```json or ```) and last line (```)
            if lines[-1].strip() == "```":
                lines = lines[1:-1]
            else:
                lines = lines[1:]
            raw = "\n".join(lines)

        data = json.loads(raw)

        song_title = data.get("song_title", "Untitled")
        style_prompt = data.get("style_prompt", "")
        lyrics = data.get("lyrics", "")

        # Build settings display
        w = data.get("weirdness", "N/A")
        w_reason = data.get("weirdness_reasoning", "")
        si = data.get("style_influence", "N/A")
        si_reason = data.get("style_influence_reasoning", "")
        settings = f"Weirdness: {w}/100\n{w_reason}\n\nStyle Influence: {si}/100\n{si_reason}"

        cover_art = data.get("cover_art_prompt", "")

        return song_title, style_prompt, lyrics, settings, cover_art

    except json.JSONDecodeError:
        # If JSON parsing fails, show raw response
        return (
            "",
            f"[JSON parse error - raw response below]\n\n{raw}",
            "",
            "",
            "",
        )
    except Exception as e:
        return "", f"Error: {e}", "", "", ""


def toggle_custom_visibility(choice):
    """Show/hide custom model text field."""
    return gr.update(visible=(choice == "Custom"))


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────

theme = gr.themes.Base(
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

with gr.Blocks(title="Suno Prompt Generator") as app:
    gr.Markdown("# Suno Prompt Generator\nDescribe your song idea in natural language. Get back structured Suno prompts.")

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
        label="Song Idea",
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

    outputs = [title_output, style_output, lyrics_output, settings_output, cover_art_output]

    generate_btn.click(
        fn=generate_prompt,
        inputs=[song_input, model_dropdown, custom_model_input, weirdness_slider],
        outputs=outputs,
    )

    song_input.submit(
        fn=generate_prompt,
        inputs=[song_input, model_dropdown, custom_model_input, weirdness_slider],
        outputs=outputs,
    )

if __name__ == "__main__":
    app.launch(inbrowser=True, theme=theme)
