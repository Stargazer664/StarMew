import backend.evaluation as evaluation
from backend.evaluation import RAGASEvaluator


def test_evaluate_maps_ragas_factual_correctness_mode_suffix(monkeypatch):
    """Ragas 0.4.x 的 factual_correctness 列名带 mode 后缀时仍应被识别。"""
    monkeypatch.setattr(evaluation, "_RAGAS_AVAILABLE", True)
    evaluator = RAGASEvaluator()

    def fake_ragas(**kwargs):
        return {
            "faithfulness": 0.9,
            "answer_relevancy": 0.8,
            "context_precision": 0.7,
            "context_recall": 0.6,
            "factual_correctness(mode=f1)": 0.73,
        }

    evaluator._evaluate_with_ragas = fake_ragas
    result = evaluator.evaluate(
        question="问题",
        answer="回答",
        retrieved_docs=[{"text": "上下文"}],
        ground_truth="标准答案",
    )

    assert result["metrics"]["factual_correctness"] == 0.73


def test_get_ragas_llm_wraps_langchain_model_for_compatible_endpoint(monkeypatch):
    """兼容 OpenAI 协议的自定义端点应避开会卡住的 InstructorLLM。"""
    import langchain.chat_models as chat_models
    import ragas.llms as ragas_llms

    raw_llm = object()
    wrapped_llm = object()
    init_kwargs = {}

    def fake_init_chat_model(**kwargs):
        init_kwargs.update(kwargs)
        return raw_llm

    def fake_wrapper(model, **kwargs):
        assert model is raw_llm
        assert kwargs == {"bypass_n": True}
        return wrapped_llm

    monkeypatch.setattr(evaluation, "_RAGAS_AVAILABLE", True)
    monkeypatch.setattr(evaluation, "EVAL_MODEL", "judge-model")
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.delenv("EVALUATION_PROVIDER", raising=False)
    monkeypatch.setenv("OLLAMA_API_KEY", "ollama")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
    monkeypatch.setattr(chat_models, "init_chat_model", fake_init_chat_model)
    monkeypatch.setattr(ragas_llms, "LangchainLLMWrapper", fake_wrapper)
    monkeypatch.setattr(
        ragas_llms,
        "llm_factory",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("不应使用 InstructorLLM")),
    )

    result = RAGASEvaluator()._get_ragas_llm()

    assert result is wrapped_llm
    assert init_kwargs == {
        "model": "judge-model",
        "model_provider": "openai",
        "api_key": "ollama",
        "base_url": "http://127.0.0.1:11434/v1",
        "temperature": 0,
        "max_retries": 0,
        "reasoning_effort": "none",
    }


def test_calculate_overall_score_includes_new_factual_score():
    """断点补算 factual_correctness 后应按全部有效指标重新计算综合分。"""
    calculate = getattr(evaluation, "calculate_overall_score", None)

    assert callable(calculate), "缺少可复用的综合分计算函数"
    assert calculate({
        "faithfulness": 0.9,
        "answer_relevancy": 0.8,
        "context_precision": 0.7,
        "context_recall": 0.6,
        "factual_correctness": 0.5,
    }) == 0.7083
