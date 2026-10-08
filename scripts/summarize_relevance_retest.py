"""Pure offline summary of the fixed-threshold, 60-question retrieval retest.

The raw baseline and shadow top10 are projections of this same measurement.
They are not a previous reranker-off experiment or extra production calls.
Only standard-library code and the existing offline validators are imported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import summarize_evidence_accuracy as base


INPUTS = ('measurement', 'dataset', 'corpus', 'qrels')
FAILURE_STATUSES = {'rerank_failed', 'infrastructure_failure', 'budget_exhausted', 'duplicate_query'}
VIEWS = {'raw_top5': 5, 'filtered_top5': 5, 'raw_top10': 10, 'shadow_filtered_top10': 10}
LOSS_REASONS = ('kept', 'not_recalled', 'unscored', 'below_threshold', 'top_k_limit',
                'retrieval_unavailable', 'processing_failure')


def _number(value, context):
    base._require(type(value) in (int, float) and math.isfinite(value), f'{context}: 必须是有限数值')
    return value


def _validate_filter(row, raw, kept, protocol):
    """Reconstruct decisions independently from fresh, input-indexed scores."""
    meta = row['meta']
    policy = meta.get('relevance_filter')
    failure = (row['status'] == 'exception' or meta.get('retrieval_mode') == 'failed'
               or meta.get('retrieval_status') in FAILURE_STATUSES)
    if policy is None:
        base._require(failure and not kept, 'measurement: 正常召回必须有 relevance_filter；失败不得返回 docs')
        return {}, [], failure, None
    base._require(isinstance(policy, dict), 'relevance_filter: 必须是对象')
    base._require(policy.get('enabled') is True, 'relevance_filter: 本协议必须启用阈值')
    base._require(policy.get('score_field') == protocol['score_field'], 'relevance_filter: score_field 不匹配')
    base._require(_number(policy.get('threshold'), 'relevance_filter.threshold') == protocol['threshold'],
                  'relevance_filter: threshold 不匹配')
    candidates = base._rows(policy.get('candidates'), 'relevance_filter.candidates')
    base._require(len(candidates) == len(raw), 'relevance_filter: 必须包含每个原始候选的判定')
    scores, by_index = {}, {}
    for candidate in candidates:
        index = candidate.get('index')
        base._require(type(index) is int and 0 <= index < len(raw) and index not in by_index,
                      'relevance_filter: index 越界/重复/类型错误')
        by_index[index] = candidate
        original = raw[index]
        base._require(candidate.get('chunk_id') == original['chunk_id'], 'relevance_filter: index/chunk_id 不匹配')
        if 'filename' in candidate:
            base._require(candidate['filename'] == original['filename'], 'relevance_filter: filename 不匹配')
        base._hash_matches(candidate.get('text_sha256'), hashlib.sha256(original['text'].encode('utf-8')).hexdigest(),
                           'relevance_filter.text_sha256')
        base._require('score' in candidate, 'relevance_filter: 缺少 score（未评分应为 null）')
        score = candidate['score']
        if score is not None:
            _number(score, 'relevance_filter.score')
            base._require(protocol['score_field'] != 'llm_rerank_score' or 0 <= score <= 10,
                          'relevance_filter: LLM 分数超出 [0,10]')
            scores[index] = score
    ranked = sorted(scores, key=lambda i: (-scores[i], i))
    accepted = [i for i in ranked if scores[i] >= protocol['threshold']]
    expected_kept = accepted[:protocol['top_k']]
    counts = {'input_count': len(raw), 'scored_count': len(scores), 'kept_count': len(expected_kept),
              'rejected_count': len(scores) - len(accepted), 'unscored_count': len(raw) - len(scores),
              'top_k_omitted_count': len(accepted) - len(expected_kept)}
    for key, expected in counts.items():
        base._require(type(policy.get(key)) is int and policy[key] == expected, f'relevance_filter.{key}: 计数不一致')
    for index, candidate in by_index.items():
        expected = ('unscored' if index not in scores else 'kept' if index in expected_kept else
                    'top_k_limit' if index in accepted else 'below_threshold')
        base._require(candidate.get('decision') == expected, 'relevance_filter: decision 与分数/顺序不一致')
    expected_status = ('no_candidates' if not raw else 'passed' if expected_kept else
                       'scoring_failed' if not scores else 'incomplete' if len(scores) < len(raw) else 'filtered_all')
    base._require(policy.get('status') == expected_status, 'relevance_filter: status 与实际评分不一致')
    failed_filter = expected_status in ('scoring_failed', 'incomplete')
    if failed_filter:
        base._require(meta.get('retrieval_status') == 'rerank_failed', 'relevance_filter: 评分失败必须标记 rerank_failed')
    if expected_status == 'filtered_all':
        base._require(meta.get('retrieval_status') == 'below_threshold', 'relevance_filter: 全低分必须标记 below_threshold')
    failure = failure or failed_filter
    base._require(not failure or not kept, 'measurement: 失败不得返回已认证 docs')
    base._require(len(kept) == len(expected_kept), 'measurement: docs 数量不等于真实 kept 数量')
    for doc, index in zip(kept, expected_kept):
        original = raw[index]
        base._require(all(doc.get(key) == original.get(key) for key in ('chunk_id', 'filename', 'text')),
                      'measurement: docs 身份或顺序与 kept 不匹配')
        value = _number(doc.get(protocol['score_field']), 'measurement.docs.score')
        base._require(value == scores[index], 'measurement: docs 分数与当次评分不一致')
    return scores, accepted, failure, expected_status


def _validate_measurement(measurement, samples, chunks):
    base._require(measurement.get('status') == 'complete', 'measurement: 必须先完成全部测量')
    protocol = measurement.get('protocol')
    base._require(isinstance(protocol, dict), 'measurement: 缺少 protocol')
    base._require(type(protocol.get('top_k')) is int and protocol['top_k'] == 5, 'protocol: 本轮生产 top_k 必须为 5')
    base._require(_number(protocol.get('threshold'), 'protocol.threshold') == 6, 'protocol: 本轮固定阈值必须为 6')
    base._require(protocol.get('score_field') in ('llm_rerank_score', 'rerank_score'), 'protocol: 未知 score_field')
    candidate_k = protocol.get('candidate_k', 30)
    base._require(type(candidate_k) is int and candidate_k == 30, 'protocol: 本轮候选上限必须为 30')
    records = base._unique(base._rows(measurement.get('records'), 'measurement.records'), ('sample_id',), 'measurement')
    base._require(set(records) == set(samples), 'measurement: 必须恰好包含全部 60 道题，不能漏题或增题')
    validated = {}
    for identifier, row in records.items():
        sample = samples[identifier]
        base._require(row.get('question') == sample['question'], 'measurement: question 与冻结题目不匹配')
        base._require(row.get('answerability') == sample['metadata']['answerability'], 'measurement: answerability 不匹配')
        base._require(row.get('status') in ('success', 'exception'), 'measurement: 未完成或未知 status')
        base._require(type(row.get('raw_available')) is bool, 'measurement: raw_available 必须明确为 bool')
        base._require(isinstance(row.get('meta'), dict), 'measurement: meta 必须是对象')
        base._require(_number(row.get('elapsed_seconds'), 'measurement.elapsed_seconds') >= 0, 'measurement: 延迟不可为负数')
        if row['status'] == 'exception':
            base._text(row.get('error_type'), 'measurement.error_type')
        raw, kept = base._rows(row.get('raw_docs'), 'raw_docs'), base._rows(row.get('docs'), 'docs')
        base._require(len(raw) <= candidate_k and len(kept) <= protocol['top_k'], 'measurement: 返回数超出协议')
        base._require(row['raw_available'] or not raw, 'measurement: raw_available=false 却存在 raw_docs')
        for doc in raw + kept:
            base._validate_candidate(doc, chunks, f'measurement.{identifier}')
        scores, accepted, failure, filter_status = _validate_filter(row, raw, kept, protocol)
        base._require(row['raw_available'] or failure, 'measurement: 正常评分缺少原始召回记录')
        validated[identifier] = {'row': row, 'scores': scores, 'accepted': accepted,
                                 'failure': failure, 'filter_status': filter_status}
    return validated


def _metrics(relevant, ids, k):
    hits, returned = len(set(ids) & relevant), len(ids)
    precision = hits / returned if returned else 0.0
    recall = hits / len(relevant)
    return {'hits': hits, 'returned_count': returned, 'precision_at_k': hits / k, 'recall_at_k': recall,
            'actual_precision': precision, 'f1_actual': 2 * precision * recall / (precision + recall) if hits else 0.0,
            'duplicate_count': returned - len(set(ids)), 'chunk_ids': ids}


def _aggregate(questions, view, k):
    rows = [q['views'][view] for q in questions]
    nonempty = [row for row in rows if row['returned_count']]
    returned = sum(row['returned_count'] for row in rows)
    return {'k': k, 'question_count': len(rows),
            'precision_at_k_macro': statistics.mean(r['precision_at_k'] for r in rows),
            'recall_at_k_macro': statistics.mean(r['recall_at_k'] for r in rows),
            'actual_precision_micro': sum(r['hits'] for r in rows) / returned if returned else 0.0,
            'actual_precision_macro': statistics.mean(r['actual_precision'] for r in rows),
            'actual_precision_nonempty_macro': statistics.mean(r['actual_precision'] for r in nonempty) if nonempty else None,
            'f1_actual_macro': statistics.mean(r['f1_actual'] for r in rows),
            'mean_returned_count': returned / len(rows), 'total_returned_count': returned,
            'nonempty_count': len(nonempty), 'empty_count': len(rows) - len(nonempty),
            'failure_count': sum(q['failure'] for q in questions),
            'incomplete_count': sum(q['filter_status'] == 'incomplete' for q in questions),
            'scoring_failed_count': sum(q['filter_status'] == 'scoring_failed' for q in questions),
            'filtered_all_count': sum(q['filter_status'] == 'filtered_all' for q in questions),
            'raw_unavailable_count': sum(not q['raw_available'] for q in questions),
            'dense_fallback_count': sum(q['retrieval_mode'] == 'dense_fallback' for q in questions)}


def _losses(relevant, raw, kept_ids, scores, accepted, available):
    groups = {key: [] for key in LOSS_REASONS}
    raw_ids, kept_ids = [d['chunk_id'] for d in raw], set(kept_ids)
    for identifier in sorted(relevant):
        positions = [i for i, value in enumerate(raw_ids) if value == identifier]
        scored = [i for i in positions if i in scores]
        if identifier in kept_ids:
            reason = 'kept'
        elif not available:
            reason = 'retrieval_unavailable'
        elif not positions:
            reason = 'not_recalled'
        elif not scored:
            reason = 'unscored'
        elif any(i in accepted[:5] for i in positions):
            reason = 'processing_failure'
        elif any(i in accepted for i in positions):
            reason = 'top_k_limit'
        else:
            reason = 'below_threshold'
        groups[reason].append(identifier)
    return groups


def summarize(*, measurement_path, dataset_path, corpus_path, qrels_path):
    paths = {'measurement': measurement_path, 'dataset': dataset_path, 'corpus': corpus_path, 'qrels': qrels_path}
    values, hashes = {}, {}
    for name, path in paths.items():
        values[name], hashes[name] = base._read(Path(path))
    dataset, corpus, qrels, measurement = (values[name] for name in ('dataset', 'corpus', 'qrels', 'measurement'))
    for name in ('dataset', 'corpus'):
        base._hash_matches(qrels.get(name + '_sha256'), hashes[name], 'qrels.' + name)
    for name in ('dataset', 'corpus', 'qrels'):
        base._hash_matches(measurement.get(name + '_sha256'), hashes[name], 'measurement.' + name)
    samples, chunks = base._validate_dataset(dataset), base._validate_corpus(corpus)
    labels = base._validate_qrels(qrels, samples, chunks)
    base._require(len(samples) == 60 and len(labels) == 50, '本轮要求 60 道题：50 可答、10 不可答')
    base._require(len(chunks) == 1351, '本轮要求冻结的 1351 个 L3 chunk')
    base._require(all(v['scope_complete'] and v['reference_facts_verified'] for v in labels.values()),
                  'qrels: 所有标签必须完成来源范围及参考事实核验')
    validated = _validate_measurement(measurement, samples, chunks)
    questions = []
    for identifier, sample in samples.items():
        item = validated[identifier]
        row, scores, accepted = item['row'], item['scores'], item['accepted']
        raw = row['raw_docs']
        filtered = [] if item['failure'] else [d['chunk_id'] for d in row['docs']]
        shadow = [] if item['failure'] else [raw[i]['chunk_id'] for i in accepted[:10]]
        question = {'sample_id': identifier, 'question': sample['question'], 'answerability': row['answerability'],
                    'status': row['status'], 'failure': item['failure'], 'error_type': row.get('error_type'),
                    'retrieval_status': row['meta'].get('retrieval_status'),
                    'retrieval_mode': row['meta'].get('retrieval_mode'), 'filter_status': item['filter_status'],
                    'raw_available': row['raw_available'], 'raw_count': len(raw), 'scored_count': len(scores),
                    'unscored_count': len(raw) - len(scores), 'elapsed_seconds': row['elapsed_seconds'],
                    'returned_count': len(filtered), 'empty': not filtered}
        if row['answerability'] == 'answerable':
            relevant = set(labels[identifier]['relevant_chunk_ids'])
            view_ids = {'raw_top5': [d['chunk_id'] for d in raw[:5]], 'filtered_top5': filtered,
                        'raw_top10': [d['chunk_id'] for d in raw[:10]], 'shadow_filtered_top10': shadow}
            question['views'] = {view: _metrics(relevant, view_ids[view], k) for view, k in VIEWS.items()}
            question['relevant_count'] = len(relevant)
            question['loss_attribution'] = _losses(relevant, raw, filtered, scores, accepted, row['raw_available'])
        questions.append(question)
    positive = [q for q in questions if q['answerability'] == 'answerable']
    negative = [q for q in questions if q['answerability'] == 'unanswerable']
    raw_count, scored_count = sum(q['raw_count'] for q in questions), sum(q['scored_count'] for q in questions)
    losses = {key: sum(len(q['loss_attribution'][key]) for q in positive) for key in LOSS_REASONS}
    limitations = [
        '固定阈值 6、生产 top_k=5；50 道可答题始终进入宏平均分母。异常、评分失败和无结果的过滤指标计零。',
        'raw 是本次同一生产调用实际召回的候选顺序；即便后续评分失败，已取得 raw 仍照实计诊断指标。不是旧 reranker-off A/B。',
        'shadow top10 从同次有效分数 >=6、分数降序/原始位置打破平分构造；没有再次调用生产 top_k=10。',
        '固定 P@K=唯一相关命中/K；actual P=唯一相关命中/实际返回条数。重复位置不压缩、不重复增加命中。',
        'actual P 宏平均对空结果计零；微平均按总命中/总返回计算。nonempty 条件宏只诊断非空题，不能替代主表。',
        'F1 逐题使用 actual precision 与 Recall 的调和平均，再按全部 50 题宏平均；不是回答事实 F1 或 Accuracy。',
        'AI source-scoped qrels，非人工金标准；仅题中限定来源的证据算相关，其余来源按题意不相关。',
        'Recall 分母为题目相关的唯一 L3 chunk；重叠分块分别计数，不是独立事实覆盖率。题集已曝光且探索性，不是正式留出集。',
        '失败、empty 和 dense_fallback 是可重叠计数。dense_fallback 表示真实路径降级，不能声称全部为纯 Hybrid 成功。',
        'incomplete 表示已评分候选均低于阈值且仍有未评分候选，不等于 API/连接错误；scoring_failed 表示没有有效评分；filtered_all 要求全部实际候选均完成评分且低于阈值。failure 为流程不可确认结果的总计，含 incomplete。',
        '未获取 raw 的失败单列 retrieval_unavailable，不当成证据未召回。流失归因以每题唯一相关 chunk 为单位，先看是否被保留。',
        '评分覆盖率分母仅为实际观察到的 raw 候选位置；未召回和召回不可用不算已经被评分为负。',
        '10 道不可答题仅报告检索结果分布；空检索不等于正确拒答，本次未生成回答，因此不提供 Accuracy/拒答率。',
    ]
    return {'schema_version': 1, 'complete': True, 'created_at': datetime.now(timezone.utc).isoformat(),
            'input_sha256': hashes, 'input_paths': {k: str(Path(p).resolve()) for k, p in paths.items()},
            'protocol': measurement['protocol'], 'human_reviewed': False, 'review_method': 'ai_source_checked',
            'audit': {'question_count': 60, 'answerable_count': 50, 'unanswerable_count': 10, 'corpus_chunk_count': 1351,
                      'failure_count': sum(q['failure'] for q in questions),
                      'exception_count': sum(q['status'] == 'exception' for q in questions),
                      'filter_status_counts': dict(Counter(q['filter_status'] or 'unavailable' for q in questions)),
                      'retrieval_mode_counts': dict(Counter(q['retrieval_mode'] or 'unavailable' for q in questions)),
                      'dense_fallback_count': sum(q['retrieval_mode'] == 'dense_fallback' for q in questions)},
            'answerable': {view: _aggregate(positive, view, k) for view, k in VIEWS.items()},
            'unanswerable': {'question_count': 10, 'empty_count': sum(q['empty'] for q in negative),
                            'filtered_all_count': sum(q['filter_status'] == 'filtered_all' and not q['failure'] for q in negative),
                            'incomplete_count': sum(q['filter_status'] == 'incomplete' for q in negative),
                            'scoring_failed_count': sum(q['filter_status'] == 'scoring_failed' for q in negative),
                            'failure_count': sum(q['failure'] for q in negative),
                            'returned_count': sum(not q['empty'] for q in negative),
                            'total_returned_chunks': sum(q['returned_count'] for q in negative)},
            'scoring_coverage': {'raw_count': raw_count, 'scored_count': scored_count,
                                 'unscored_count': raw_count - scored_count,
                                 'scored_fraction': scored_count / raw_count if raw_count else None,
                                 'unscored_fraction': (raw_count - scored_count) / raw_count if raw_count else None,
                                 'raw_available_questions': sum(q['raw_available'] for q in questions)},
            'relevant_evidence_attribution': {'unit': 'question_unique_relevant_chunk', 'counts': losses,
                                            'total': sum(losses.values())},
            'questions': questions, 'limitations': limitations}


def _format(value):
    return '不可用' if value is None else f'{value:.4f}'


def _markdown(report):
    lines = ['# 固定阈值全题检索重测', '', '阈值 6，生产 top 5；50 可答题 + 10 不可答题。AI 来源核验标签，非人工签核。',
             '', '| 对照 | 题数 | 固定 P@K | Recall@K | actual P 微 | actual P 宏 | 非空 P 宏 | F1 宏 | 平均返回 | 空 | 失败 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for view, row in report['answerable'].items():
        metrics = ' | '.join(_format(row[key]) for key in ('precision_at_k_macro', 'recall_at_k_macro',
            'actual_precision_micro', 'actual_precision_macro', 'actual_precision_nonempty_macro', 'f1_actual_macro',
            'mean_returned_count'))
        lines.append(f"| {view} | {row['question_count']} | {metrics} | {row['empty_count']} | {row['failure_count']} |")
    no, coverage = report['unanswerable'], report['scoring_coverage']
    lines += ['', f"不可答题：空结果 {no['empty_count']}/10；全低分 {no['filtered_all_count']}/10；"
              f"流程不可确认 {no['failure_count']}/10；有返回 {no['returned_count']}/10（共 {no['total_returned_chunks']} 条）。",
              f"其中 incomplete {no['incomplete_count']}；scoring_failed {no['scoring_failed_count']}；两者与全低分分别报告。",
              f"dense_fallback 降级：{report['audit']['dense_fallback_count']}/60。",
              f"评分覆盖：{coverage['scored_count']}/{coverage['raw_count']} = {_format(coverage['scored_fraction'])}；"
              f"未评分 {coverage['unscored_count']}。", '', '相关证据归因（题目 × 唯一相关 chunk）：', '']
    lines += [f'- {key}: {count}' for key, count in report['relevant_evidence_attribution']['counts'].items()]
    lines += ['', '口径与限制：', ''] + ['- ' + item for item in report['limitations']]
    lines += ['', '输入 SHA256：', ''] + [f'- {key}: `{value}`' for key, value in report['input_sha256'].items()]
    return '\n'.join(lines) + '\n'


def write_report(report, output_dir):
    output_dir = Path(output_dir)
    base._require(not output_dir.exists(), f'输出目录已存在，拒绝覆盖: {output_dir}')
    summary = {key: value for key, value in report.items() if key != 'questions'}
    artifacts = {'summary.json': json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                 'summary.md': _markdown(report),
                 'questions.json': json.dumps({'input_sha256': report['input_sha256'], 'questions': report['questions']},
                                              ensure_ascii=False, indent=2, allow_nan=False) + '\n'}
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, content in artifacts.items():
        with (output_dir / name).open('x', encoding='utf-8', newline='') as handle:
            handle.write(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in INPUTS:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    try:
        base._require(not args.output_dir.exists(), f'输出目录已存在，拒绝覆盖: {args.output_dir}')
        report = summarize(**{name + '_path': getattr(args, name) for name in INPUTS})
        write_report(report, args.output_dir)
    except (ValueError, OSError) as exc:
        parser.exit(2, f'离线汇总失败: {exc}\n')
    print(f'昕，离线汇总完成：{args.output_dir}（全部 60 题，失败保留分母）')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
