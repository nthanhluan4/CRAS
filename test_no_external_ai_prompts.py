from pathlib import Path


def test_no_google_or_chatgpt_prompts_in_app_source():
    app_src = Path("app.py").read_text(encoding="utf-8")
    vlm_src = Path("vlm_inspector.py").read_text(encoding="utf-8")

    assert "Google Gemini" not in app_src
    assert "ChatGPT" not in app_src
    assert "gemini_api_key_store" not in app_src
    assert "Google Gemini" not in vlm_src
    assert "ChatGPT" not in vlm_src
