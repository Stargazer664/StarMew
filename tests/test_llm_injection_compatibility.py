import backend.rag_pipeline as rag_pipeline
import backend.rag_utils as rag_utils


def test_rag_pipeline_preserves_model_factory_injection(monkeypatch):
    marker = object()
    monkeypatch.setattr(rag_pipeline, "_grader_model", None)
    monkeypatch.setattr(rag_pipeline, "init_chat_model", lambda **kwargs: marker)

    assert rag_pipeline._get_grader_model() is marker


def test_rag_utils_preserves_model_factory_injection(monkeypatch):
    marker = object()
    monkeypatch.setattr(rag_utils, "_stepback_model", None)
    monkeypatch.setattr(rag_utils, "init_chat_model", lambda **kwargs: marker)

    assert rag_utils._get_stepback_model() is marker
