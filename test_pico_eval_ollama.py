"""The legacy Pico evaluation entry point uses the shared local LLM settings."""

from scripts import run_pico_eval


def test_pico_pipeline_uses_shared_generation_model_factory(monkeypatch):
    captured = {}
    marker = object()

    def fake_create_chat_model(**kwargs):
        captured.update(kwargs)
        return marker

    monkeypatch.setattr(run_pico_eval, "create_chat_model", fake_create_chat_model)

    pipeline = run_pico_eval.create_eval_pipeline()

    assert callable(pipeline)
    assert captured == {"role": "generation", "temperature": 0}
