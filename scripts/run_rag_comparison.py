"""Command line runner for controlled RAG/no-RAG paired evaluations."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.eval_comparison.contracts import (  # noqa: E402
    ComparisonConfig,
    ComparisonSample,
    GenerationResult,
    MetricResult,
    load_samples,
    save_frozen_samples,
    validate_samples,
)
from backend.eval_comparison.generation import (  # noqa: E402
    COMMON_SYSTEM_PROMPT,
    RetrievalProcess,
    create_generation_llm,
    generate_answer,
)
from backend.eval_comparison.reporting import export_report, summarize_pairs  # noqa: E402
from backend.eval_comparison.scoring import RagasScorer, score_answer, score_rag_context  # noqa: E402
from backend.eval_comparison.storage import (  # noqa: E402
    CheckpointWriter,
    RunLock,
    canonical_fingerprint,
    load_resume,
    save_checkpoint,
    utc_now,
)
from backend.llm_config import resolve_llm_settings  # noqa: E402


SOURCE_FILES = [
    "backend/eval_comparison/contracts.py",
    "backend/eval_comparison/generation.py",
    "backend/eval_comparison/scoring.py",
    "backend/eval_comparison/storage.py",
    "backend/eval_comparison/reporting.py",
    "scripts/run_rag_comparison.py",
    "uv.lock",
]


def _real_retrieval_target(question: str, _config: dict[str, Any]) -> dict[str, Any]:
    from backend.rag_pipeline import run_rag_graph

    return run_rag_graph(question)


class PipelineDependencies:
    """Injection boundary for offline CLI tests and real service adapters."""

    def create_llm(self, config: ComparisonConfig):
        return create_generation_llm(config)

    def create_retriever(self, config: ComparisonConfig):
        return RetrievalProcess(_real_retrieval_target)

    def create_scorer(self, config: ComparisonConfig):
        return RagasScorer(judge_model=config.judge_model, timeout=config.scoring_timeout)

    async def generate(
        self,
        sample: ComparisonSample,
        arm: str,
        config: ComparisonConfig,
        llm: Any,
        retriever: Any,
    ) -> GenerationResult:
        result = await generate_answer(
            sample.question,
            arm=arm,
            config=config,
            llm=llm,
            retriever=retriever,
        )
        result.sample_id = sample.id
        return result

    async def score(
        self,
        sample: ComparisonSample,
        answer: str,
        scorer: Any,
        metrics: list[str],
        timeout: float,
    ) -> dict[str, MetricResult]:
        return await score_answer(
            sample,
            answer,
            scorer=scorer,
            metrics=metrics,
            timeout=timeout,
        )

    async def diagnose(
        self,
        sample: ComparisonSample,
        answer: str,
        contexts: list[str],
        scorer: Any,
        metrics: list[str],
        timeout: float,
    ) -> dict[str, MetricResult]:
        return await score_rag_context(
            sample,
            answer,
            used_contexts=contexts,
            scorer=scorer,
            metrics=metrics,
            timeout=timeout,
        )

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _endpoint_identifier() -> str | None:
    value = resolve_llm_settings(role="generation").base_url
    if not value:
        return None
    parsed = urlsplit(value)
    return f"{parsed.scheme}://{parsed.netloc}" if parsed.netloc else "configured-custom-endpoint"


def _llm_identity() -> dict[str, Any]:
    generation = resolve_llm_settings(role="generation")
    judge = resolve_llm_settings(role="evaluation")
    return {
        "provider": generation.provider,
        "generation_base_url": generation.base_url,
        "judge_provider": judge.provider,
        "judge_base_url": judge.base_url,
    }


def _git_identity() -> dict[str, Any]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return completed.stdout.strip()

    return {
        "head": run("rev-parse", "HEAD") or None,
        "branch": run("branch", "--show-current") or None,
        "dirty": bool(run("status", "--porcelain")),
    }


def _source_hashes() -> dict[str, str | None]:
    return {name: _sha256(PROJECT_ROOT / name) for name in SOURCE_FILES}


def _corpus_identity() -> dict[str, Any]:
    documents = PROJECT_ROOT / "data" / "documents"
    entries: list[dict[str, Any]] = []
    if documents.is_dir():
        for path in sorted(item for item in documents.rglob("*") if item.is_file()):
            entries.append(
                {
                    "path": path.relative_to(PROJECT_ROOT).as_posix(),
                    "sha256": _sha256(path),
                    "size": path.stat().st_size,
                }
            )
    return {
        "bm25_state_sha256": _sha256(PROJECT_ROOT / "data" / "bm25_state.json"),
        "documents": entries,
        "reproducibility_limited": not entries,
    }


def _ragas_version() -> str | None:
    try:
        import ragas

        return str(ragas.__version__)
    except Exception:
        return None


def _select_samples(
    samples: list[ComparisonSample], *, split: str | None, limit: int | None, seed: int
) -> list[ComparisonSample]:
    selected = [sample for sample in samples if split is None or sample.metadata.split == split]
    if limit is None or limit >= len(selected):
        return selected
    grouped: dict[str, list[ComparisonSample]] = {}
    for sample in selected:
        grouped.setdefault(sample.metadata.question_type, []).append(sample)
    rng = random.Random(seed)
    for values in grouped.values():
        rng.shuffle(values)
    ordered: list[ComparisonSample] = []
    keys = sorted(grouped)
    while len(ordered) < limit and keys:
        next_keys: list[str] = []
        for key in keys:
            if grouped[key] and len(ordered) < limit:
                ordered.append(grouped[key].pop())
            if grouped[key]:
                next_keys.append(key)
        keys = next_keys
    return ordered


def _build_plan(samples: list[ComparisonSample], repeat: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    plan: list[dict[str, Any]] = []
    for sample in shuffled:
        for repeat_index in range(repeat):
            arms = ["no_rag", "rag"]
            rng.shuffle(arms)
            for arm in arms:
                plan.append(
                    {
                        "sample_id": sample.id,
                        "repeat_index": repeat_index,
                        "arm": arm,
                    }
                )
    return plan


def _new_record(item: dict[str, Any], metrics: list[str]) -> dict[str, Any]:
    return {
        **item,
        "generation_status": "pending",
        "answer": None,
        "error": None,
        "retrieved_docs": [],
        "used_contexts": [],
        "rag_trace": [],
        "retrieval_empty": None,
        "timing": {
            "retrieval_seconds": None,
            "generation_seconds": None,
            "response_seconds": None,
            "evaluation_seconds": None,
        },
        "metrics": {name: None for name in metrics},
        "metric_status": {name: "pending" for name in metrics},
        "metric_errors": {},
        "metric_reasons": {},
        "usage": {
            "generation_input_tokens": None,
            "generation_output_tokens": None,
            "retrieval_llm_tokens": None,
        },
        "generation_attempts": [],
        "scoring_attempts": [],
    }


def _config_from_args(args: argparse.Namespace) -> ComparisonConfig:
    model = args.model or resolve_llm_settings(role="generation").model
    judge = args.judge_model or resolve_llm_settings(role="evaluation").model
    return ComparisonConfig(
        model=model,
        judge_model=judge,
        repeat=args.repeat,
        seed=args.seed,
        score_concurrency=args.score_concurrency,
        metrics=[name.strip() for name in args.metrics.split(",") if name.strip()],
        diagnostics=[name.strip() for name in args.diagnostics.split(",") if name.strip()],
        top_k=args.top_k,
        max_context_chars=args.max_context_chars,
        generation_timeout=args.generation_timeout,
        stream=args.stream,
        retrieval_timeout=args.retrieval_timeout,
        scoring_timeout=args.scoring_timeout,
        max_attempts=args.max_attempts,
    )


def _fingerprint_inputs(dataset_path: Path, config: ComparisonConfig) -> dict[str, Any]:
    return {
        "dataset_sha256": _sha256(dataset_path),
        "config": config.model_dump(mode="json"),
        "prompt_sha256": hashlib.sha256(COMMON_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "source_hashes": _source_hashes(),
        "corpus_identity": _corpus_identity(),
        "ragas_version": _ragas_version(),
        "llm_identity": _llm_identity(),
    }


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


async def _with_heartbeat(awaitable: Any, label: str) -> Any:
    task = asyncio.ensure_future(awaitable)
    started = time.perf_counter()
    while True:
        done, _ = await asyncio.wait({task}, timeout=30)
        if task in done:
            return await task
        print(f"[等待] {label} 已等待 {time.perf_counter() - started:.0f}s", flush=True)


def _is_retryable_error(error: str | None) -> bool:
    if not error:
        return False
    lowered = error.lower()
    return any(
        marker in lowered
        for marker in (
            "timeout",
            "apiconnectionerror",
            "connection error",
            "network",
            "429",
            "rate limit",
            "500",
            "502",
            "503",
            "504",
        )
    )


def _retry_delay(config: ComparisonConfig, item: dict[str, Any], attempt_number: int) -> float:
    jitter = random.Random(
        f"{config.seed}:{item['sample_id']}:{item['repeat_index']}:"
        f"{item['arm']}:{attempt_number}"
    ).random()
    return 2 ** attempt_number + jitter


def _record_map(payload: dict[str, Any]) -> dict[tuple[str, int, str], dict[str, Any]]:
    return {
        (record["sample_id"], int(record["repeat_index"]), record["arm"]): record
        for record in payload["records"]
    }


async def _generate_phase(
    manifest: dict[str, Any],
    payload: dict[str, Any],
    samples: dict[str, ComparisonSample],
    writer: CheckpointWriter,
    deps: PipelineDependencies,
    *,
    retry_failed: bool,
) -> None:
    config = ComparisonConfig.model_validate(manifest["config"])
    records = _record_map(payload)
    eligible = [
        item for item in manifest["plan"]
        if records[(item["sample_id"], item["repeat_index"], item["arm"])]["generation_status"] == "pending"
        or (
            retry_failed
            and records[(item["sample_id"], item["repeat_index"], item["arm"])]["generation_status"] == "failed"
        )
    ]
    if not eligible:
        return
    llm = deps.create_llm(config)
    retriever = deps.create_retriever(config) if any(item["arm"] == "rag" for item in eligible) else None
    try:
        for position, item in enumerate(eligible, start=1):
            key = (item["sample_id"], item["repeat_index"], item["arm"])
            record = records[key]
            if len(record.get("generation_attempts", [])) >= config.max_attempts:
                continue
            started = time.perf_counter()
            while len(record.get("generation_attempts", [])) < config.max_attempts:
                attempt_started = time.perf_counter()
                record["generation_status"] = "running"
                writer.upsert(record)
                result = await _with_heartbeat(
                    deps.generate(
                        samples[item["sample_id"]], item["arm"], config, llm, retriever
                    ),
                    f"生成 sample={item['sample_id']} arm={item['arm']}",
                )
                preserved = {
                    "metrics": record["metrics"],
                    "metric_status": record["metric_status"],
                    "metric_errors": record["metric_errors"],
                    "metric_reasons": record.get("metric_reasons", {}),
                    "scoring_attempts": record.get("scoring_attempts", []),
                }
                updated = result.model_dump(mode="json")
                updated.update(item)
                updated.update(preserved)
                updated["generation_attempts"] = record.get("generation_attempts", []) + [
                    {
                        "status": result.generation_status,
                        "error": result.error,
                        "elapsed_seconds": time.perf_counter() - attempt_started,
                        "finished_at": utc_now(),
                    }
                ]
                record = updated
                records[key] = updated
                writer.upsert(updated)
                attempt_number = len(updated["generation_attempts"])
                if (
                    result.generation_status == "success"
                    or attempt_number >= config.max_attempts
                    or not _is_retryable_error(result.error)
                ):
                    break
                delay = _retry_delay(config, item, attempt_number)
                print(
                    f"[生成重试] sample={item['sample_id']} arm={item['arm']} "
                    f"第 {attempt_number + 1}/{config.max_attempts} 次将在 {delay:.1f}s 后开始",
                    flush=True,
                )
                await deps.sleep(delay)
            print(
                f"[生成 {position}/{len(eligible)}] sample={item['sample_id']} "
                f"arm={item['arm']} 耗时={time.perf_counter() - started:.1f}s 已保存",
                flush=True,
            )
    finally:
        if hasattr(retriever, "close"):
            retriever.close()


async def _score_phase(
    manifest: dict[str, Any],
    payload: dict[str, Any],
    samples: dict[str, ComparisonSample],
    writer: CheckpointWriter,
    deps: PipelineDependencies,
    *,
    retry_failed: bool,
) -> None:
    config = ComparisonConfig.model_validate(manifest["config"])
    records = _record_map(payload)
    eligible: list[dict[str, Any]] = []
    for item in manifest["plan"]:
        record = records[(item["sample_id"], item["repeat_index"], item["arm"])]
        if record["generation_status"] != "success":
            continue
        statuses = record.get("metric_status", {})
        if any(status == "pending" or (retry_failed and status == "failed") for status in statuses.values()):
            eligible.append(item)
    if not eligible:
        return
    scorer = deps.create_scorer(config)
    semaphore = asyncio.Semaphore(config.score_concurrency)

    async def score_item(item: dict[str, Any]) -> tuple[dict[str, Any], float]:
        key = (item["sample_id"], item["repeat_index"], item["arm"])
        record = records[key]
        selected_metrics = [
            name for name, status in record["metric_status"].items()
            if status == "pending" or (retry_failed and status == "failed")
        ]
        started = time.perf_counter()
        async with semaphore:
            pending_metrics = list(selected_metrics)
            while pending_metrics:
                for name in pending_metrics:
                    record["metric_status"][name] = "running"
                writer.upsert(record)
                results = await _with_heartbeat(
                    deps.score(
                        samples[item["sample_id"]],
                        record["answer"],
                        scorer,
                        pending_metrics,
                        config.scoring_timeout,
                    ),
                    f"评分 sample={item['sample_id']} arm={item['arm']}",
                )
                for name in pending_metrics:
                    result = results.get(name) or MetricResult(
                        name=name, status="not_applicable", reason="scorer returned no result"
                    )
                    record["metrics"][name] = result.value
                    record["metric_status"][name] = result.status
                    if result.error:
                        record["metric_errors"][name] = result.error
                    else:
                        record["metric_errors"].pop(name, None)
                    if result.reason:
                        record.setdefault("metric_reasons", {})[name] = result.reason
                record.setdefault("scoring_attempts", []).append(
                    {
                        "metrics": list(pending_metrics),
                        "status": "success"
                        if all(
                            record["metric_status"][name] in {"success", "not_applicable"}
                            for name in pending_metrics
                        )
                        else "failed",
                        "errors": {
                            name: record["metric_errors"].get(name)
                            for name in pending_metrics
                            if record["metric_errors"].get(name)
                        },
                        "finished_at": utc_now(),
                    }
                )
                writer.upsert(record)

                retry_names: list[str] = []
                for name in pending_metrics:
                    attempts = sum(
                        name in attempt.get("metrics", [])
                        for attempt in record.get("scoring_attempts", [])
                    )
                    if (
                        record["metric_status"][name] == "failed"
                        and attempts < config.max_attempts
                        and _is_retryable_error(record["metric_errors"].get(name))
                    ):
                        retry_names.append(name)
                if not retry_names:
                    break
                attempt_number = max(
                    sum(
                        name in attempt.get("metrics", [])
                        for attempt in record.get("scoring_attempts", [])
                    )
                    for name in retry_names
                )
                delay = _retry_delay(config, item, attempt_number)
                await deps.sleep(delay)
                pending_metrics = retry_names

            if item["arm"] == "rag" and config.diagnostics:
                contexts = [context.get("text", "") for context in record.get("used_contexts", [])]
                diagnostic_results = await deps.diagnose(
                    samples[item["sample_id"]],
                    record["answer"],
                    contexts,
                    scorer,
                    config.diagnostics,
                    config.scoring_timeout,
                )
                for name, result in diagnostic_results.items():
                    record["metrics"][name] = result.value
                    record["metric_status"][name] = result.status
                    if result.error:
                        record["metric_errors"][name] = result.error
                    if result.reason:
                        record.setdefault("metric_reasons", {})[name] = result.reason
        for name in selected_metrics:
            if name not in record["metrics"]:
                result = MetricResult(
                    name=name, status="not_applicable", reason="scorer returned no result"
                )
                record["metrics"][name] = result.value
                record["metric_status"][name] = result.status
        elapsed = time.perf_counter() - started
        record["timing"]["evaluation_seconds"] = elapsed
        if record.get("scoring_attempts"):
            record["scoring_attempts"][-1]["total_elapsed_seconds"] = elapsed
        records[key] = record
        writer.upsert(record)
        return item, elapsed

    tasks = [asyncio.create_task(score_item(item)) for item in eligible]
    completed = 0
    for future in asyncio.as_completed(tasks):
        item, elapsed = await future
        completed += 1
        print(
            f"[评分 {completed}/{len(eligible)}] sample={item['sample_id']} "
            f"arm={item['arm']} 耗时={elapsed:.1f}s 已保存",
            flush=True,
        )


def _phase_exit_code(payload: dict[str, Any], phase: str) -> int:
    records = payload["records"]
    if phase in {"generate", "all"} and any(
        record["generation_status"] != "success" for record in records
    ):
        return 1
    if phase in {"score", "all"}:
        for record in records:
            if record["generation_status"] != "success":
                return 1
            if any(
                status not in {"success", "not_applicable"}
                for status in record.get("metric_status", {}).values()
            ):
                return 1
    return 0


def _run_async(coro: Any) -> Any:
    return asyncio.run(coro)


def _create_run(args: argparse.Namespace, deps: PipelineDependencies) -> int:
    run_dir = args.run_dir.resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        print(f"参数错误：run_dir 已存在且非空: {run_dir}", file=sys.stderr)
        return 2
    run_dir.mkdir(parents=True, exist_ok=True)
    with RunLock(run_dir):
        samples = load_samples(args.dataset)
        errors = validate_samples(samples, formal=args.formal)
        if errors:
            for index, error in enumerate(errors, start=1):
                print(f"[校验 {index}/{len(errors)}] {error}", file=sys.stderr)
            return 2
        selected = _select_samples(samples, split=args.split, limit=args.limit, seed=args.seed)
        if not selected:
            print("参数错误：没有匹配选定 split 的样本", file=sys.stderr)
            return 2
        config = _config_from_args(args)
        if not config.model or not config.judge_model:
            print("参数错误：缺少生成模型或裁判模型", file=sys.stderr)
            return 2
        dataset_path = run_dir / "dataset.json"
        save_frozen_samples(dataset_path, selected, source=str(args.dataset.resolve()))
        fingerprint_inputs = _fingerprint_inputs(dataset_path, config)
        fingerprint = canonical_fingerprint(fingerprint_inputs)
        plan = _build_plan(selected, config.repeat, config.seed)
        run_id = run_dir.name
        manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "created_at": utc_now(),
            "dataset_source": str(args.dataset.resolve()),
            "dataset_sha256": fingerprint_inputs["dataset_sha256"],
            "git": _git_identity(),
            "source_hashes": fingerprint_inputs["source_hashes"],
            "uv_lock_sha256": _sha256(PROJECT_ROOT / "uv.lock"),
            "ragas_version": fingerprint_inputs["ragas_version"],
            "endpoint": _endpoint_identifier(),
            "llm_provider": fingerprint_inputs["llm_identity"]["provider"],
            "actual_generation_model": config.model,
            "actual_judge_model": config.judge_model,
            "prompt_sha256": fingerprint_inputs["prompt_sha256"],
            "corpus_identity": fingerprint_inputs["corpus_identity"],
            "config": config.model_dump(mode="json"),
            "selected_sample_ids": [sample.id for sample in selected],
            "plan": plan,
            "fingerprint_inputs": fingerprint_inputs,
            "fingerprint": fingerprint,
        }
        _write_manifest(run_dir / "manifest.json", manifest)
        payload = {
            "schema_version": 1,
            "run_id": run_id,
            "fingerprint": fingerprint,
            "updated_at": utc_now(),
            "records": [_new_record(item, config.metrics) for item in plan],
        }
        writer = CheckpointWriter(run_dir, payload)
        sample_map = {sample.id: sample for sample in selected}
        if args.phase in {"generate", "all"}:
            _run_async(_generate_phase(manifest, writer.snapshot(), sample_map, writer, deps, retry_failed=False))
        if args.phase in {"score", "all"}:
            _run_async(_score_phase(manifest, writer.snapshot(), sample_map, writer, deps, retry_failed=False))
        final = writer.snapshot()
        if args.phase == "all":
            export_report(run_dir, summarize_pairs(final["records"], selected, seed=config.seed), final["records"])
        return _phase_exit_code(final, args.phase)


def _load_run(
    run_dir: Path, *, verify_fingerprint: bool = True
) -> tuple[dict[str, Any], list[ComparisonSample], dict[str, Any]]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    samples = load_samples(run_dir / "dataset.json")
    if not verify_fingerprint:
        payload = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))
        return manifest, samples, payload
    current_inputs = deepcopy(manifest["fingerprint_inputs"])
    current_inputs["dataset_sha256"] = _sha256(run_dir / "dataset.json")
    current_inputs["source_hashes"] = _source_hashes()
    current_inputs["corpus_identity"] = _corpus_identity()
    current_inputs["llm_identity"] = _llm_identity()
    expected = canonical_fingerprint(current_inputs)
    payload = load_resume(run_dir, expected)
    return manifest, samples, payload


def _resume(args: argparse.Namespace, deps: PipelineDependencies) -> int:
    run_dir = args.run_dir.resolve()
    with RunLock(run_dir):
        manifest, samples, payload = _load_run(run_dir)
        writer = CheckpointWriter(run_dir, payload)
        sample_map = {sample.id: sample for sample in samples}
        if args.phase in {"generate", "all"}:
            _run_async(
                _generate_phase(
                    manifest, writer.snapshot(), sample_map, writer, deps,
                    retry_failed=args.retry_failed,
                )
            )
        if args.phase in {"score", "all"}:
            _run_async(
                _score_phase(
                    manifest, writer.snapshot(), sample_map, writer, deps,
                    retry_failed=args.retry_failed,
                )
            )
        final = writer.snapshot()
        if args.phase == "all":
            config = ComparisonConfig.model_validate(manifest["config"])
            export_report(
                run_dir,
                summarize_pairs(final["records"], samples, seed=config.seed),
                final["records"],
            )
        return _phase_exit_code(final, args.phase)


def _validate(args: argparse.Namespace) -> int:
    samples = load_samples(args.dataset)
    errors = validate_samples(samples, formal=args.formal)
    for index, error in enumerate(errors, start=1):
        print(f"[校验 {index}/{len(errors)}] {error}")
    save_frozen_samples(args.output, samples, source=str(args.dataset.resolve()))
    print(f"已加载 {len(samples)} 条，结构问题 {len(errors)} 条；已保存 {args.output}")
    return 0 if not errors else 2


def _report(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.resolve()
    manifest, samples, payload = _load_run(run_dir, verify_fingerprint=False)
    config = ComparisonConfig.model_validate(manifest["config"])
    summary = summarize_pairs(payload["records"], samples, seed=config.seed)
    export_report(run_dir, summary, payload["records"])
    print(f"报告已保存: {run_dir / 'report.md'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG 与无 RAG 配对测评")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="转换并校验数据集")
    validate.add_argument("--dataset", type=Path, required=True)
    validate.add_argument("--output", type=Path, required=True)
    validate.add_argument("--formal", action="store_true")

    run = subparsers.add_parser("run", help="创建新运行")
    run.add_argument("--dataset", type=Path, required=True)
    run.add_argument("--split", choices=["dev", "test", "exploratory"])
    run.add_argument("--limit", type=int)
    run.add_argument("--repeat", type=int, default=1)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--metrics", default="factual_correctness,abstention")
    run.add_argument("--diagnostics", default="")
    run.add_argument("--score-concurrency", type=int, default=2)
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--phase", choices=["generate", "all"], default="all")
    run.add_argument("--formal", action="store_true")
    run.add_argument("--model")
    run.add_argument("--judge-model")
    run.add_argument("--top-k", type=int, default=5)
    run.add_argument("--max-context-chars", type=int, default=8000)
    run.add_argument("--generation-timeout", type=float, default=180)
    run.add_argument("--stream", action="store_true", help="测量真实流式首回答内容时间")
    run.add_argument("--retrieval-timeout", type=float, default=300)
    run.add_argument("--scoring-timeout", type=float, default=180)
    run.add_argument("--max-attempts", type=int, default=3)

    resume = subparsers.add_parser("resume", help="按 manifest 续跑")
    resume.add_argument("--run-dir", type=Path, required=True)
    resume.add_argument("--phase", choices=["generate", "score", "all"], default="all")
    resume.add_argument("--retry-failed", action="store_true")

    report = subparsers.add_parser("report", help="离线重算报告")
    report.add_argument("--run-dir", type=Path, required=True)
    return parser


def main(
    argv: list[str] | None = None,
    *,
    dependencies: PipelineDependencies | None = None,
) -> int:
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
        args = build_parser().parse_args(argv)
        deps = dependencies or PipelineDependencies()
        if args.command == "validate":
            return _validate(args)
        if args.command == "run":
            return _create_run(args, deps)
        if args.command == "resume":
            return _resume(args, deps)
        if args.command == "report":
            return _report(args)
        return 2
    except (FileNotFoundError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"参数或指纹错误: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
