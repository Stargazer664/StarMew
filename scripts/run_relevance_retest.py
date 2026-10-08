"""Measure the fixed threshold on the complete frozen set using live retrieval.

Read-only index/model inference; no answer generation or threshold selection.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BENCHMARK = ROOT / 'data/resume_benchmark_20261004'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def save_new(path, value):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ['NO_PROXY'] = '127.0.0.1,localhost,::1'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'

    from backend import rag_utils as rag
    from backend.agent_budget import AgentBudget, AgentRunState, use_run_state
    from backend.conversation_context import model_budget
    from scripts.summarize_evidence_accuracy import (
        _validate_dataset, _validate_corpus, _validate_qrels, _validate_candidate,
    )
    settings = rag.resolve_llm_settings(role='generation')
    endpoint = urlsplit(settings.base_url)
    manager = rag._milvus_manager
    local = ('127.0.0.1', 'localhost', '::1')
    assert settings.provider == 'ollama' and endpoint.hostname in local
    assert not endpoint.username and not endpoint.password and endpoint.scheme in ('http', 'https')
    assert manager.host in local
    policy = rag._relevance_filter_policy()
    assert policy['enabled'] and policy['threshold'] == 6 and policy['score_field'] == 'llm_rerank_score'
    assert rag.LEAF_RETRIEVE_LEVEL == 3 and rag.RETRIEVAL_CANDIDATE_MULTIPLIER == 6
    inputs = {'dataset': BENCHMARK / 'dataset_reference_v2.json',
              'corpus': BENCHMARK / 'evidence_accuracy_v1/corpus_snapshot.json',
              'qrels': BENCHMARK / 'evidence_accuracy_v1/qrels.json'}
    frozen, hashes = {}, {}
    for key, path in inputs.items():
        raw = path.read_bytes()
        hashes[key] = digest(raw)
        (output / (key + '.json')).write_bytes(raw)
        frozen[key] = json.loads(raw)
    assert frozen['qrels']['dataset_sha256'] == hashes['dataset']
    assert frozen['qrels']['corpus_sha256'] == hashes['corpus']
    samples = _validate_dataset(frozen['dataset'])
    corpus = _validate_corpus(frozen['corpus'])
    _validate_qrels(frozen['qrels'], samples, corpus)
    assert len(samples) == 60 and len(frozen['qrels']['samples']) == 50
    assert manager.collection_name == frozen['corpus']['collection']

    # Deny accidental Python HTTP connections outside the checked local services.
    # gRPC uses native sockets; its configured Milvus endpoint was checked above.
    ports = {endpoint.port or (443 if endpoint.scheme == 'https' else 80), int(manager.port)}
    original_connect, original_connect_ex = socket.socket.connect, socket.socket.connect_ex
    def guard(address):
        if not isinstance(address, tuple) or len(address) < 2 or address[0] not in local or address[1] not in ports:
            raise RuntimeError('Only checked local inference/index endpoints are permitted')
    def connect(sock, address):
        guard(address)
        return original_connect(sock, address)
    def connect_ex(sock, address):
        guard(address)
        return original_connect_ex(sock, address)
    socket.socket.connect, socket.socket.connect_ex = connect, connect_ex

    budget = AgentBudget.from_env()
    source_files = ['scripts/run_relevance_retest.py', 'backend/rag_utils.py', 'backend/milvus_client.py',
                    'backend/embedding.py', 'backend/bm25.py', 'backend/agent_budget.py',
                    'backend/llm_config.py', 'backend/conversation_context.py',
                    'backend/evidence_context.py', 'backend/chat_diagnostics.py']
    source_hashes = {name: digest((ROOT / name).read_bytes()) for name in source_files}
    protocol = {'created_at': datetime.now(timezone.utc).isoformat(), 'top_k': 5, 'candidate_k': 30,
                'threshold': 6, 'score_field': 'llm_rerank_score', 'model': settings.model,
                'provider': settings.provider, 'collection': manager.collection_name,
                'agent_budget': asdict(budget), 'context_budget': asdict(model_budget()),
                'sample_order': list(samples), 'per_question_attempts': 1,
                'scope': 'Live initial Hybrid L3 retrieval plus current rerank/filter; no grader, rewrite, parent merge, or answer generation.',
                'baseline': 'Same live raw30 candidate list, truncated at 5/10. Not old reranker-off production.',
                'shadow_at_10': 'Derived from the same scores; not a production top_k=10 call.',
                'labels': 'Frozen AI source-scoped qrels, not human-reviewed; known exploratory set.',
                'failures': 'Keep all 50 answerable questions in denominator; negatives are separate.',
                'process_only_overrides': {'NO_PROXY': os.environ['NO_PROXY'], 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'},
                'input_sha256': hashes, 'source_sha256': source_hashes}
    save_new(output / 'protocol.json', protocol)

    fields = ['id', 'chunk_id', 'filename', 'text', 'chunk_level', 'file_type',
              'parent_chunk_id', 'root_chunk_id', 'page_number', 'chunk_idx']
    def read_corpus():
        def read():
            rag.assert_index_available()
            manager.assert_bm25_compatible()
            return manager.query_all('chunk_level == 3', output_fields=fields)
        with use_run_state(AgentRunState(budget=AgentBudget(total_timeout_seconds=60, final_reserve_seconds=0))):
            rows = rag.run_serialized_index_mutation(read)
        assert len(rows) == len(corpus)
        assert {d['chunk_id'] for d in rows} == set(corpus)
        for doc in rows:
            _validate_candidate(doc, corpus, 'live index')
        return sorted(rows, key=lambda d: d['chunk_id'])

    print('昕，[准备] 核验现有索引与冻结标签的全文、来源和 ID。', flush=True)
    before = read_corpus()
    save_new(output / 'live_corpus_before.json', {'chunks': before})
    print(f'昕，✓ 索引 {len(before)} 个 L3 全部匹配；开始本地嵌入预热。', flush=True)
    warmup = time.perf_counter()
    rag._embedding_service.get_embeddings(['检索连通性预热'])
    warmup_seconds = time.perf_counter() - warmup
    print(f'昕，✓ 嵌入预热完成（{warmup_seconds:.2f}s），固定阈值 6 开始测量。', flush=True)

    measurement = {'status': 'running', 'protocol': protocol, 'records': [],
                   'dataset_sha256': hashes['dataset'], 'corpus_sha256': hashes['corpus'],
                   'qrels_sha256': hashes['qrels'], 'warmup_seconds': warmup_seconds}
    def checkpoint():
        tmp = output / 'measurement.json.tmp'
        tmp.write_text(json.dumps(measurement, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        tmp.replace(output / 'measurement.json')
    checkpoint()
    captured = []
    original_snapshot = rag._snapshot_candidates
    def capture(*a, **kw):
        snapshot = original_snapshot(*a, **kw)
        captured.append(deepcopy(snapshot['docs']))
        return snapshot
    rag._snapshot_candidates = capture
    try:
        for number, (identifier, sample) in enumerate(samples.items(), 1):
            print(f'昕，[{number}/60] 全库召回并评分 {identifier}', flush=True)
            captured.clear()
            row = {'sample_id': identifier, 'question': sample['question'],
                   'answerability': sample['metadata']['answerability'], 'status': 'success',
                   'raw_docs': [], 'docs': [], 'meta': {}, 'error_type': None}
            started = time.perf_counter()
            try:
                with use_run_state(AgentRunState(budget=budget)):
                    result = rag.retrieve_documents(sample['question'], top_k=5)
                row.update(docs=result['docs'], meta=result['meta'])
            except Exception as exc:
                row.update(status='exception', error_type=type(exc).__name__)
            row['raw_docs'] = captured[-1] if captured else []
            row['raw_available'] = bool(captured)
            row['elapsed_seconds'] = time.perf_counter() - started
            row['snapshot_count'] = len(captured)
            assert len(captured) <= 1
            for doc in row['raw_docs'] + row['docs']:
                _validate_candidate(doc, corpus, identifier)
            save_new(output / (identifier + '.json'), row)
            measurement['records'].append(row)
            checkpoint()
            filt = row['meta'].get('relevance_filter', {})
            print(f"昕，✓ [{number}/60] 保留 {len(row['docs'])}，评分 {filt.get('scored_count', 0)}/{len(row['raw_docs'])}，状态 {filt.get('status', row['status'])}，{row['elapsed_seconds']:.2f}s", flush=True)
    finally:
        rag._snapshot_candidates = original_snapshot
    after = read_corpus()
    save_new(output / 'live_corpus_after.json', {'chunks': after})
    assert before == after, 'Index changed during measurement'
    assert all(digest(p.read_bytes()) == hashes[k] for k, p in inputs.items())
    assert all(digest((ROOT / k).read_bytes()) == v for k, v in source_hashes.items())
    measurement.update(status='complete', completed_at=datetime.now(timezone.utc).isoformat(),
                       inputs_unchanged=True, index_unchanged=True, source_unchanged=True)
    checkpoint()
    manager._reset_client()
    print('昕，✓ 60/60 测量完成，索引、标签及业务源码未变。', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
