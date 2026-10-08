"""Ensure every model_provider configuration uses LangChain's native OpenAI provider."""

from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PROVIDER_CUSTOM = re.compile(
    r"model_provider\s*=\s*(?:\\?[\"'])openai_custom(?:\\?[\"'])"
)


def test_no_model_provider_openai_custom_literals_remain():
    """Executable Python/notebook configurations must use native openai."""
    offenders = []
    for path in PROJECT_ROOT.rglob("*"):
        if path.suffix not in {".py", ".ipynb"} or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if MODEL_PROVIDER_CUSTOM.search(text):
            offenders.append(path.relative_to(PROJECT_ROOT).as_posix())

    assert offenders == [], (
        "Found model_provider=" + repr("openai_custom") + " in: " + ", ".join(sorted(offenders))
    )
