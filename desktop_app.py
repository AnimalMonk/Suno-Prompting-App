"""
Desktop entry point for packaged Suno Prompt Generator.
Launches Gradio and opens it in the default browser.
Runs without a visible console window (PyInstaller --windowed).
"""

import multiprocessing
import os
import signal
import sys
import webbrowser


def main():
    # Determine the install/exe directory for .env loading
    if getattr(sys, "frozen", False):
        # Running as a PyInstaller bundle
        app_dir = os.path.dirname(sys.executable)
    else:
        # Running as a script (dev mode)
        app_dir = os.path.dirname(os.path.abspath(__file__))

    # Load .env from the install directory
    env_path = os.path.join(app_dir, ".env")
    from dotenv import load_dotenv
    load_dotenv(env_path)

    # Import and build the Gradio app
    from app import create_app
    demo, theme = create_app()

    # Launch Gradio — opens in default browser, blocks until closed
    demo.launch(
        server_name="127.0.0.1",
        server_port=0,  # Random free port
        share=False,
        inbrowser=True,
        theme=theme,
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
