# PyInstaller hook for Gradio.
# Gradio reads its own .py source files at runtime for component metadata,
# so we must collect them as source rather than compiled bytecode.

module_collection_mode = {
    "gradio": "py",
}
