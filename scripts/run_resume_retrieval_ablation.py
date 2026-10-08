"""Read-only dense/BM25 hybrid retrieval comparison; no LLM calls or index writes."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
import time
from contextlib import nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def collect_retrieval(samples, embed, manager, *, seed=42, progress=None, checkpoint=None, call_context=nullcontext):
    rng = random.Random(seed)
    records = []
    selected = [s for s in samples if s['metadata']['answerability'] == 'answerable' and s.get('expected_docs')]
    rng.shuffle(selected)
    for index, sample in enumerate(selected, 1):
        if progress:
            progress(index, len(selected), sample['id'])
        started = time.perf_counter()
        try:
            vector = embed([sample['question']])[0]
            embedding_error = None
        except Exception as exc:
            vector, embedding_error = None, type(exc).__name__
        embedding_seconds = time.perf_counter() - started
        modes = ['dense', 'hybrid']
        rng.shuffle(modes)
        for mode in modes:
            record = {'sample_id':sample['id'], 'mode':mode, 'status':'failed', 'docs':None,
                      'embedding_seconds':embedding_seconds, 'search_seconds':None, 'error_type':embedding_error}
            started = time.perf_counter()
            if embedding_error is None:
                try:
                    kwargs = {'top_k':10, 'filter_expr':'chunk_level == 3'}
                    if mode == 'hybrid':
                        kwargs['query_text'] = sample['question']
                    with call_context():
                        docs = getattr(manager, mode + '_retrieve')(vector, **kwargs)
                    record.update(status='success', docs=docs)
                except Exception as exc:
                    record['error_type'] = type(exc).__name__
                record['search_seconds'] = time.perf_counter() - started
            records.append(record)
            if checkpoint:
                checkpoint(records)
    return records


def summarize_ablation(samples, records):
    from scripts.summarize_resume_metrics import evaluate_retrieval
    expected = {s['id']:s.get('expected_docs', []) for s in samples}
    indexed = {}
    failed = 0
    unscorable = 0
    for record in records:
        if record['status'] != 'success':
            failed += 1
            continue
        score = evaluate_retrieval(expected[record['sample_id']], record['docs'])
        if score['status'] != 'success':
            unscorable += 1
            continue
        indexed[(record['sample_id'], record['mode'])] = score
    paired = sorted(sid for sid in expected if (sid, 'dense') in indexed and (sid, 'hybrid') in indexed)
    keys = ['coverage_at_5', 'hit_at_5', 'coverage_at_10', 'hit_at_10', 'mrr_at_10']
    modes = {mode:{key:statistics.fmean(indexed[(sid,mode)][key] for sid in paired) if paired else None
                   for key in keys} for mode in ['dense','hybrid']}
    return {'paired_questions':len(paired), 'failed_records':failed, 'unscorable_records':unscorable,
            'modes':modes,
            'hybrid_minus_dense':{key:modes['hybrid'][key]-modes['dense'][key] if paired else None for key in keys},
            'metric_scope':'预期来源覆盖率与来源命中率，非穷尽相关文档标注的 Recall。排名保留原始 chunk 位次。'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    if args.output.suffix.lower() != '.json':
        parser.error('output must end with .json')
    dataset_snapshot = args.output.with_suffix('.dataset.json')
    if any(p.exists() for p in [args.output, args.output.with_suffix('.md'), dataset_snapshot, args.output.with_suffix('.json.tmp')]):
        parser.error('output or companion file already exists; choose a new path')
    from backend.eval_comparison.contracts import load_samples, validate_samples
    samples = load_samples(args.dataset)
    errors = validate_samples(samples, formal=False)
    if errors:
        parser.error('; '.join(errors))
    if not any(s.metadata.answerability == 'answerable' and s.expected_docs for s in samples):
        parser.error('no answerable questions with expected_docs')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Freeze bytes before loading models; always evaluate this frozen copy.
    dataset_bytes = args.dataset.read_bytes()
    dataset_snapshot.write_bytes(dataset_bytes)
    samples = load_samples(dataset_snapshot)
    errors = validate_samples(samples, formal=False)
    if errors:
        parser.error('frozen dataset: ' + '; '.join(errors))
    if not any(s.metadata.answerability == 'answerable' and s.expected_docs for s in samples):
        parser.error('frozen dataset has no answerable questions with expected_docs')
    source_hashes = {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
        ['scripts/run_resume_retrieval_ablation.py','scripts/summarize_resume_metrics.py',
         'backend/milvus_client.py','backend/embedding.py','backend/bm25.py']}
    from backend.embedding import embedding_service
    from backend.milvus_client import MilvusManager
    from backend.agent_budget import AgentBudget, AgentRunState, use_run_state
    manager = MilvusManager()
    payload = {'measurement_kind':'retrieval_only', 'seed':args.seed, 'status':'running',
        'dataset_sha256':hashlib.sha256(dataset_bytes).hexdigest(), 'source_sha256':source_hashes,
        'collection':manager.collection_name, 'top_k':10, 'records':[],
        'scope':'真实现有索引的叶子检索；dense与hybrid共享每题同一个embedding；不含重排、父块合并、查询重写、生成或HTTP。'}
    def save_checkpoint(records):
        payload['records'] = records
        temporary = args.output.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        temporary.replace(args.output)
    save_checkpoint([])
    # Warm-up uses a separate fixed connectivity query, not a benchmark question.
    started = time.perf_counter()
    embedding_service.get_embeddings(['检索连通性预热'])
    warmup_seconds = time.perf_counter() - started
    print(f'昕，嵌入预热完成，耗时 {warmup_seconds:.2f}s', flush=True)
    rows = [s.model_dump(mode='json') for s in samples]
    try:
        records = collect_retrieval(rows, embedding_service.get_embeddings, manager, seed=args.seed,
            progress=lambda i,n,sid: print(f'昕，[{i}/{n}] 检索消融 {sid}', flush=True),
            checkpoint=save_checkpoint,
            call_context=lambda:use_run_state(AgentRunState(budget=AgentBudget(total_timeout_seconds=35,final_reserve_seconds=5))))
    finally:
        manager._reset_client()
    summary = summarize_ablation(rows, records)
    payload.update(status='complete',warmup_seconds=warmup_seconds,summary=summary)
    save_checkpoint(records)
    lines = ['# 检索消融实测', '', payload['scope'], '', summary['metric_scope'], '',
             f"有效配对题数：{summary['paired_questions']}；失败记录：{summary['failed_records']}；不可评分：{summary['unscorable_records']}。",
             '', '| 指标 | Dense | Hybrid | Hybrid − Dense |', '|---|---:|---:|---:|']
    for key in summary['modes']['dense']:
        values = [summary['modes']['dense'][key], summary['modes']['hybrid'][key], summary['hybrid_minus_dense'][key]]
        lines.append('| '+key+' | '+' | '.join(f'{v:.4f}' if v is not None else '未获得' for v in values)+' |')
    lines.extend(['', '单轮、单机、AI 证据校对题集；不代表人类审核测试集或生产总体效果。',
                  '两组共用每题的同一个向量，检索顺序按固定种子随机。原始 JSON 保存每条候选与阶段耗时。'])
    args.output.with_suffix('.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(f'昕，已保存 {len(records)} 条原始检索记录：{args.output}', flush=True)
    return 0 if all(r['status']=='success' for r in records) else 1


if __name__ == '__main__':
    raise SystemExit(main())
