"""Offline consumer regressions; no model, database or frozen result is used."""
import ast
import asyncio
import importlib
import io
import json
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console
from rich.table import Table


ROOT = Path(__file__).resolve().parent


@pytest.fixture
def resume(monkeypatch):
    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **kw: False)
    return importlib.import_module("scripts.resume_dishes_factual_eval")


def test_resume_summary_empty_scores_are_unknown_not_zero(resume):
    payload = {"evaluations": [{"metrics": {}, "overall_score": None}]}
    resume.update_summary(payload)
    summary = payload["summary"]
    assert summary["avg_overall_score"] is None
    assert summary["avg_factual_correctness"] is None
    assert summary["valid_overall_count"] == 0
    assert summary["valid_counts"]["factual_correctness"] == 0


def test_resume_summary_excludes_failures_but_keeps_zero_and_legacy_valid(resume):
    rows = [
        {"metrics": {"faithfulness": 0}, "overall_score": 0},
        {"metrics": {"faithfulness": 1}, "overall_score": 1,
         "metric_status": {"faithfulness": "success"}},
        {"metrics": {"faithfulness": .5}, "overall_score": .5, "error": "SDK failed"},
        {"metrics": {"faithfulness": .5}, "overall_score": .5,
         "metric_status": {"faithfulness": "failed"}},
        {"metrics": {"faithfulness": .5}, "overall_score": .5,
         "metric_status": {"faithfulness": "not_applicable"}},
        {"metrics": {"faithfulness": .5}, "overall_score": .5,
         "metric_errors": {"faithfulness": "timeout"}},
    ]
    payload = {"evaluations": deepcopy(rows)}
    resume.update_summary(payload)
    assert payload["summary"]["avg_faithfulness"] == .5
    assert payload["summary"]["valid_counts"]["faithfulness"] == 2
    assert payload["summary"]["avg_overall_score"] == .5
    assert payload["summary"]["valid_overall_count"] == 2
    assert payload["evaluations"] == rows


@pytest.mark.parametrize("invalid", [True, "0.5", float("nan"), float("inf"), -.01, 1.01])
def test_resume_summary_invalid_numeric_values_do_not_become_completed(resume, invalid):
    payload = {"evaluations": [{"metrics": {"factual_correctness": invalid}, "overall_score": invalid}]}
    resume.update_summary(payload)
    assert payload["summary"]["avg_factual_correctness"] is None
    assert payload["summary"]["avg_overall_score"] is None
    assert payload["summary"]["factual_completed"] == 0


@pytest.mark.parametrize("invalid", [True, "0.5", float("nan"), float("inf"), -.01, 1.01])
def test_resume_score_one_rejects_invalid_sdk_scores(resume, invalid):
    class Metric:
        async def single_turn_ascore(self, sample):
            return invalid
    _, value, error, _ = asyncio.run(resume.score_one(
        0, {"question": "q", "answer": "a", "ground_truth": "r"}, Metric(), asyncio.Semaphore(1)))
    assert value is None
    assert error


@pytest.mark.parametrize("new_score", [.8, None])
def test_resume_run_updates_only_factual_status_and_preserves_other_failures(resume, monkeypatch, tmp_path, new_score):
    source, output = tmp_path / "source.json", tmp_path / "output.json"
    row = {"question": "q", "answer": "a", "ground_truth": "r",
           "metrics": {"faithfulness": .5, "factual_correctness": None},
           "overall_score": .5, "error": "historical batch error",
           "metric_status": {"faithfulness": "failed", "factual_correctness": "failed"},
           "metric_errors": {"faithfulness": "historical failure", "factual_correctness": "old error"}}
    source.write_text(json.dumps({"evaluations": [row]}), encoding="utf-8")
    before = source.read_bytes()
    monkeypatch.setattr(resume, "RAGASEvaluator", lambda: SimpleNamespace(_get_ragas_llm=lambda: object()))
    class Metric:
        async def single_turn_ascore(self, sample):
            if new_score is None:
                raise RuntimeError("new factual failure")
            return new_score
    monkeypatch.setattr(resume, "FactualCorrectness", lambda **kw: Metric())
    asyncio.run(resume.run(source, output, 1))
    result = json.loads(output.read_text(encoding="utf-8"))
    saved = result["evaluations"][0]
    assert saved["metric_status"]["factual_correctness"] == ("success" if new_score is not None else "failed")
    assert saved["metric_status"]["faithfulness"] == "failed"
    assert saved["metric_errors"]["faithfulness"] == "historical failure"
    assert saved["overall_score"] == new_score
    assert result["summary"]["avg_overall_score"] == new_score
    assert result["summary"]["valid_counts"]["faithfulness"] == 0
    assert source.read_bytes() == before


def test_resume_retries_numeric_failed_value_but_not_not_applicable(resume, monkeypatch, tmp_path):
    source, output = tmp_path / "source.json", tmp_path / "output.json"
    rows = [{"question": "retry", "answer": "a", "ground_truth": "r",
             "metrics": {"factual_correctness": .5}, "overall_score": .5,
             "metric_status": {"factual_correctness": "failed"}},
            {"question": "skip", "answer": "a", "metrics": {"factual_correctness": None},
             "overall_score": None, "metric_status": {"factual_correctness": "not_applicable"}}]
    source.write_text(json.dumps({"evaluations": rows}), encoding="utf-8")
    monkeypatch.setattr(resume, "RAGASEvaluator", lambda: SimpleNamespace(_get_ragas_llm=lambda: object()))
    class Metric:
        async def single_turn_ascore(self, sample):
            assert sample.user_input == "retry"
            return 0
    monkeypatch.setattr(resume, "FactualCorrectness", lambda **kw: Metric())
    asyncio.run(resume.run(source, output, 1))
    saved = json.loads(output.read_text(encoding="utf-8"))["evaluations"]
    assert saved[0]["metrics"]["factual_correctness"] == 0
    assert saved[0]["metric_status"]["factual_correctness"] == "success"
    assert saved[1] == rows[1]


@pytest.mark.parametrize("old_error,fully_known,expected_error", [
    ("one_or_more_metrics_failed", True, None),
    ("evaluation_failed:RuntimeError", True, None),
    ("historical unknown error", True, "historical unknown error"),
    ("one_or_more_metrics_failed", False, "one_or_more_metrics_failed"),
])
def test_resume_clears_only_known_aggregate_errors_after_full_recovery(
        resume, monkeypatch, tmp_path, old_error, fully_known, expected_error):
    source, output = tmp_path / "source.json", tmp_path / "output.json"
    metrics = dict.fromkeys(resume._METRICS_DEFINITION, .6)
    metrics["factual_correctness"] = None
    statuses = dict.fromkeys(resume._METRICS_DEFINITION, "success")
    statuses["factual_correctness"] = "failed"
    statuses["context_recall"] = "not_applicable"
    metrics["context_recall"] = None
    if not fully_known:
        statuses.pop("faithfulness")
    row = {"question": "q", "answer": "a", "ground_truth": "r", "metrics": metrics,
           "metric_status": statuses, "metric_errors": {"factual_correctness": "TimeoutError"},
           "error": old_error, "evaluation_status": "partial"}
    source.write_text(json.dumps({"evaluations": [row]}), encoding="utf-8")
    monkeypatch.setattr(resume, "RAGASEvaluator", lambda: SimpleNamespace(_get_ragas_llm=lambda: object()))
    class Metric:
        async def single_turn_ascore(self, sample):
            return .8
    monkeypatch.setattr(resume, "FactualCorrectness", lambda **kw: Metric())
    asyncio.run(resume.run(source, output, 1))
    saved = json.loads(output.read_text(encoding="utf-8"))["evaluations"][0]
    assert saved.get("error") == expected_error
    assert saved["evaluation_status"] == ("success" if expected_error is None else "partial")
    assert saved["metric_status"]["context_recall"] == "not_applicable"


def test_resume_score_error_never_persists_endpoint_details(resume):
    class Metric:
        async def single_turn_ascore(self, sample):
            raise RuntimeError("secret endpoint https://private.example/token")
    _, value, error, _ = asyncio.run(resume.score_one(
        0, {"question": "q", "answer": "a", "ground_truth": "r"}, Metric(), asyncio.Semaphore(1)))
    assert value is None
    assert error == "RuntimeError"


@pytest.mark.parametrize("overall,error,expected", [(None, "judge failed", "N/A"), (0, None, "0.0000")])
def test_pico_real_reporting_flow_handles_null_warmup(resume, overall, error, expected):
    # Import-time Pico code changes cwd/stdio and imports live RAG dependencies.
    # Execute its actual function definitions with only infrastructure boundaries replaced.
    path = ROOT / "scripts" / "run_pico_eval.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    sample = SimpleNamespace(question="q")
    result = SimpleNamespace(sample=sample, answer="answer", metrics={}, overall_score=overall, error=error)
    async def batch(**kwargs):
        return [result]
    stream = io.StringIO()
    namespace = {
        "console": Console(file=stream, color_system=None, width=120), "Table": Table,
        "time": time, "PROJECT_ROOT": ROOT, "_RAGAS_AVAILABLE": True, "_RAGAS_VERSION": "test",
        "EvalDatasetManager": SimpleNamespace(load_from_json=lambda path: [sample]),
        "evaluator": SimpleNamespace(clear_history=lambda: None, evaluate_rag_pipeline=lambda *a: result,
                                     evaluate_batch=batch, print_summary=lambda: None,
                                     export_results=lambda path: path, export_csv=lambda path: path),
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    namespace["ensure_document_indexed"] = lambda filename: True
    namespace["create_eval_pipeline"] = lambda: object()
    asyncio.run(namespace["run_evaluation"]())
    assert expected in stream.getvalue()
    if error:
        assert error in stream.getvalue()
