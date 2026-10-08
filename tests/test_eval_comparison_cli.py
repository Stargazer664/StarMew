import asyncio
import json
from pathlib import Path

import pytest

from backend.eval_comparison.contracts import GenerationResult, MetricResult
import scripts.run_rag_comparison as runner
from scripts.run_rag_comparison import PipelineDependencies, build_parser, main


FIXTURE = Path(__file__).parent / "fixtures" / "rag_comparison.json"


class BombDependencies(PipelineDependencies):
    def create_llm(self, config):
        raise AssertionError("model must not initialize")

    def create_retriever(self, config):
        raise AssertionError("retriever must not initialize")

    def create_scorer(self, config):
        raise AssertionError("scorer must not initialize")


class FakeDependencies(PipelineDependencies):
    def __init__(self, events, interrupt_after=None):
        self.events = events
        self.interrupt_after = interrupt_after
        self.generated = 0

    def create_llm(self, config):
        return object()

    def create_retriever(self, config):
        return object()

    def create_scorer(self, config):
        return object()

    async def generate(self, sample, arm, config, llm, retriever):
        if self.interrupt_after is not None and self.generated >= self.interrupt_after:
            raise KeyboardInterrupt()
        self.generated += 1
        self.events.append(("generate", sample.id, arm))
        return GenerationResult(
            sample_id=sample.id,
            arm=arm,
            generation_status="success",
            answer=f"{arm} answer",
            retrieved_docs=[] if arm == "no_rag" else [{"text": "evidence"}],
            used_contexts=[] if arm == "no_rag" else [{"text": "evidence"}],
            retrieval_empty=arm == "rag" and False,
            timing={
                "retrieval_seconds": 0.1 if arm == "rag" else 0.0,
                "generation_seconds": 0.2,
                "response_seconds": 0.3 if arm == "rag" else 0.2,
                "evaluation_seconds": None,
            },
            usage={},
            attempts=[],
        )

    async def score(self, sample, answer, scorer, metrics, timeout):
        self.events.append(("score", sample.id, answer))
        return {
            name: MetricResult(name=name, value=0.5, status="success", elapsed_seconds=0.1)
            for name in metrics
            if not (name == "factual_correctness" and sample.metadata.answerability == "unanswerable")
        }


class RetryScoreDependencies(FakeDependencies):
    def __init__(self, events):
        super().__init__(events)
        self.score_calls = 0
        self.delays = []

    async def score(self, sample, answer, scorer, metrics, timeout):
        self.score_calls += 1
        self.events.append(("score_attempt", self.score_calls, tuple(metrics)))
        if self.score_calls == 1:
            return {
                name: MetricResult(
                    name=name, value=None, status="failed",
                    error="APIConnectionError: temporary connection failure",
                )
                for name in metrics
            }
        return {
            name: MetricResult(name=name, value=0.7, status="success")
            for name in metrics
        }

    async def sleep(self, seconds):
        self.delays.append(seconds)


class RetryGenerationDependencies(FakeDependencies):
    def __init__(self, events):
        super().__init__(events)
        self.generation_calls = 0
        self.delays = []

    async def generate(self, sample, arm, config, llm, retriever):
        self.generation_calls += 1
        self.events.append(("generate_attempt", self.generation_calls, sample.id, arm))
        if self.generation_calls == 1:
            return GenerationResult(
                sample_id=sample.id,
                arm=arm,
                generation_status="failed",
                error="APIConnectionError: temporary connection failure",
            )
        return await super().generate(sample, arm, config, llm, retriever)

    async def sleep(self, seconds):
        self.delays.append(seconds)


def _run_args(run_dir, phase="all"):
    return [
        "run", "--dataset", str(FIXTURE), "--split", "dev", "--limit", "1",
        "--repeat", "1", "--seed", "42", "--metrics", "factual_correctness",
        "--run-dir", str(run_dir), "--phase", phase, "--model", "fake-model",
        "--judge-model", "fake-judge",
    ]


def test_validate_does_not_initialize_models(tmp_path):
    output = tmp_path / "frozen.json"
    code = main(
        ["validate", "--dataset", str(FIXTURE), "--output", str(output), "--formal"],
        dependencies=BombDependencies(),
    )
    assert code == 0
    assert json.loads(output.read_text(encoding="utf-8"))["total_samples"] == 2


def test_run_defaults_to_configured_ollama_models_despite_legacy_ark_env(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("EVALUATION_PROVIDER", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_JUDGE_MODEL", raising=False)
    monkeypatch.setenv("MODEL", "legacy-cloud-model")
    monkeypatch.setenv("GRADE_MODEL", "legacy-cloud-judge")
    args = build_parser().parse_args(
        ["run", "--dataset", str(FIXTURE), "--run-dir", str(tmp_path / "run")]
    )

    config = runner._config_from_args(args)

    assert config.model == "qwen3.5:4b"
    assert config.judge_model == "qwen3.5:4b"


def test_report_does_not_initialize_models(tmp_path):
    events = []
    run_dir = tmp_path / "run"
    assert main(_run_args(run_dir, phase="generate"), dependencies=FakeDependencies(events)) == 0
    code = main(["report", "--run-dir", str(run_dir)], dependencies=BombDependencies())
    assert code == 0
    assert (run_dir / "report.md").is_file()


def test_cloud_judge_default_and_manifest_leave_generation_local(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("EVALUATION_PROVIDER", "aliyun")
    monkeypatch.setenv("GRADE_MODEL", "qwen3.7-flash")
    monkeypatch.setenv("BASE_URL", "https://cloud.example/v1")
    monkeypatch.setenv("ARK_API_KEY", "test-key")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3.5:4b")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
    args = build_parser().parse_args(
        ["run", "--dataset", str(FIXTURE), "--run-dir", str(tmp_path / "run")]
    )
    config = runner._config_from_args(args)
    assert config.model == "qwen3.5:4b"
    assert config.judge_model == "qwen3.7-flash"
    assert runner._llm_identity() == {
        "provider": "ollama", "generation_base_url": "http://127.0.0.1:11434/v1",
        "judge_provider": "aliyun", "judge_base_url": "https://cloud.example/v1",
    }


def test_resume_rejects_switching_evaluation_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("EVALUATION_PROVIDER", "ollama")
    run_dir = tmp_path / "run"
    assert main(_run_args(run_dir, phase="generate"), dependencies=FakeDependencies([])) == 0
    monkeypatch.setenv("EVALUATION_PROVIDER", "aliyun")
    monkeypatch.setenv("GRADE_MODEL", "qwen3.7-flash")
    monkeypatch.setenv("BASE_URL", "https://cloud.example/v1")
    monkeypatch.setenv("ARK_API_KEY", "test-key")
    assert main(["resume", "--run-dir", str(run_dir), "--phase", "score"],
                dependencies=BombDependencies()) == 2


def test_offline_report_can_be_regenerated_after_reporting_code_changes(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    assert main(_run_args(run_dir, phase="generate"), dependencies=FakeDependencies([])) == 0
    monkeypatch.setattr(runner, "_source_hashes", lambda: {"changed.py": "new-hash"})

    code = main(["report", "--run-dir", str(run_dir)], dependencies=BombDependencies())

    assert code == 0
    assert (run_dir / "report.md").is_file()


def test_run_generates_all_answers_before_any_scoring(tmp_path):
    events = []
    code = main(_run_args(tmp_path / "run"), dependencies=FakeDependencies(events))
    assert code == 0
    kinds = [event[0] for event in events]
    assert kinds == ["generate", "generate", "score", "score"]


def test_interrupted_run_resumes_only_unfinished_generation(tmp_path):
    run_dir = tmp_path / "run"
    first_events = []
    with pytest.raises(KeyboardInterrupt):
        main(
            _run_args(run_dir),
            dependencies=FakeDependencies(first_events, interrupt_after=1),
        )
    checkpoint = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))
    assert sum(record["generation_status"] == "success" for record in checkpoint["records"]) == 1

    resumed_events = []
    code = main(
        ["resume", "--run-dir", str(run_dir), "--phase", "all"],
        dependencies=FakeDependencies(resumed_events),
    )
    assert code == 0
    assert [event[0] for event in resumed_events].count("generate") == 1
    assert [event[0] for event in resumed_events].count("score") == 2


def test_run_refuses_to_overwrite_existing_directory(tmp_path):
    run_dir = tmp_path / "existing"
    run_dir.mkdir()
    (run_dir / "keep.txt").write_text("keep", encoding="utf-8")
    assert main(_run_args(run_dir), dependencies=FakeDependencies([])) == 2
    assert (run_dir / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_resume_rejects_configuration_override_arguments(tmp_path):
    run_dir = tmp_path / "run"
    main(_run_args(run_dir, phase="generate"), dependencies=FakeDependencies([]))
    with pytest.raises(SystemExit):
        main(
            ["resume", "--run-dir", str(run_dir), "--model", "changed"],
            dependencies=FakeDependencies([]),
        )


def test_resume_rejects_switching_ollama_endpoint(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
    assert main(
        _run_args(run_dir, phase="generate"), dependencies=FakeDependencies([])
    ) == 0

    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:22434/v1")
    code = main(
        ["resume", "--run-dir", str(run_dir), "--phase", "generate"],
        dependencies=BombDependencies(),
    )

    assert code == 2


def test_retryable_scoring_error_retries_without_regenerating(tmp_path):
    events = []
    dependencies = RetryScoreDependencies(events)
    code = main(_run_args(tmp_path / "run"), dependencies=dependencies)
    assert code == 0
    assert dependencies.score_calls == 3  # first record twice, second record once
    assert len(dependencies.delays) == 1
    assert 2.0 <= dependencies.delays[0] < 3.0
    assert [event[0] for event in events].count("generate") == 2


def test_retryable_generation_error_retries_before_scoring(tmp_path):
    events = []
    dependencies = RetryGenerationDependencies(events)
    code = main(_run_args(tmp_path / "run"), dependencies=dependencies)
    assert code == 0
    assert dependencies.generation_calls == 3  # first record twice, second record once
    assert len(dependencies.delays) == 1
    assert 2.0 <= dependencies.delays[0] < 3.0
    assert [event[0] for event in events].count("score") == 2
