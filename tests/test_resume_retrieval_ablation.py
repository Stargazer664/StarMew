"""Behavior checks for a paired, read-only retrieval benchmark."""
from scripts.run_resume_retrieval_ablation import collect_retrieval
import json
import pytest


@pytest.mark.parametrize('name,existing', [('result.md',False),('result.json',True)])
def test_cli_refuses_ambiguous_output_or_existing_markdown(tmp_path, name, existing):
    from scripts.run_resume_retrieval_ablation import main
    if existing:
        (tmp_path/'result.md').write_text('keep me',encoding='utf-8')
    with pytest.raises(SystemExit) as exc:
        main(['--dataset',str(tmp_path/'missing.json'),'--output',str(tmp_path/name)])
    assert exc.value.code == 2
    if existing:
        assert (tmp_path/'result.md').read_text(encoding='utf-8') == 'keep me'


def test_cli_rejects_zero_eligible_questions_before_starting_services(tmp_path):
    from scripts.run_resume_retrieval_ablation import main
    dataset=tmp_path/'data.json'
    dataset.write_text(json.dumps({'samples':[{'id':'n','question':'未知?','ground_truth':'未提供',
        'expected_docs':['a.md'],'metadata':{'answerability':'unanswerable'}}]}),encoding='utf-8')
    with pytest.raises(SystemExit) as exc:
        main(['--dataset',str(dataset),'--output',str(tmp_path/'result.json')])
    assert exc.value.code == 2
    assert not (tmp_path/'result.json').exists()


def test_cli_preserves_existing_temporary_file(tmp_path):
    from scripts.run_resume_retrieval_ablation import main
    temporary=tmp_path/'result.json.tmp'
    temporary.write_text('previous measurement',encoding='utf-8')
    with pytest.raises(SystemExit) as exc:
        main(['--dataset',str(tmp_path/'missing.json'),'--output',str(tmp_path/'result.json')])
    assert exc.value.code == 2
    assert temporary.read_text(encoding='utf-8') == 'previous measurement'


def test_cli_revalidates_snapshot_when_source_changes_during_setup(tmp_path, monkeypatch):
    from scripts.run_resume_retrieval_ablation import main
    from backend.eval_comparison import contracts
    dataset=tmp_path/'data.json'
    dataset.write_text(json.dumps({'samples':[{'id':'q','question':'食材?', 'ground_truth':'鸡蛋',
        'expected_docs':['a.md'],'metadata':{'answerability':'answerable'}}]}),encoding='utf-8')
    original=contracts.load_samples
    def concurrent_change(path):
        result=original(path)
        if path == dataset:
            dataset.write_text('{"samples":[]}',encoding='utf-8')
        return result
    monkeypatch.setattr(contracts,'load_samples',concurrent_change)
    with pytest.raises(SystemExit) as exc:
        main(['--dataset',str(dataset),'--output',str(tmp_path/'result.json')])
    assert exc.value.code == 2
    assert not (tmp_path/'result.json').exists()


def test_summary_compares_only_matched_successes_and_discloses_failures():
    from scripts.run_resume_retrieval_ablation import summarize_ablation
    samples = [{'id':'a','expected_docs':['a.md']}, {'id':'b','expected_docs':['b.md']}]
    records = [
        {'sample_id':'a','mode':'dense','status':'success','docs':[{'filename':'x.md'},{'filename':'a.md'}]},
        {'sample_id':'a','mode':'hybrid','status':'success','docs':[{'filename':'a.md'}]},
        {'sample_id':'b','mode':'dense','status':'failed','docs':None},
        {'sample_id':'b','mode':'hybrid','status':'success','docs':[{'filename':'b.md'}]},
    ]
    result = summarize_ablation(samples, records)
    assert result['paired_questions'] == 1
    assert result['failed_records'] == 1
    assert result['modes']['dense']['mrr_at_10'] == 0.5
    assert result['modes']['hybrid']['mrr_at_10'] == 1.0
    assert result['hybrid_minus_dense']['mrr_at_10'] == 0.5


def test_both_modes_share_one_embedding_and_keep_full_candidate_order():
    calls = []
    def embed(texts):
        calls.append(('embed', texts))
        return [[0.5]]
    class Manager:
        def dense_retrieve(self, vector, **kwargs):
            calls.append(('dense', vector, kwargs))
            return [{'filename': 'b.md'}, {'filename': 'a.md'}]
        def hybrid_retrieve(self, vector, **kwargs):
            calls.append(('hybrid', vector, kwargs))
            return [{'filename': 'a.md'}]
    samples = [{'id':'q1', 'question':'食材?', 'expected_docs':['a.md'], 'metadata':{'answerability':'answerable'}}]
    records = collect_retrieval(samples, embed, Manager(), seed=42)
    assert sum(c[0] == 'embed' for c in calls) == 1
    assert {r['mode'] for r in records} == {'dense', 'hybrid'}
    assert all(r['status'] == 'success' for r in records)
    assert next(r for r in records if r['mode'] == 'dense')['docs'] == [{'filename':'b.md'}, {'filename':'a.md'}]
    assert all(c[2]['top_k'] == 10 and c[2]['filter_expr'] == 'chunk_level == 3' for c in calls if c[0] != 'embed')


def test_unanswerable_is_not_retrieval_ground_truth_and_failure_stays_missing():
    class Manager:
        def dense_retrieve(self, *args, **kwargs):
            raise TimeoutError('unavailable')
        def hybrid_retrieve(self, *args, **kwargs):
            return []
    samples = [
        {'id':'negative','question':'未知?', 'expected_docs':['a.md'],'metadata':{'answerability':'unanswerable'}},
        {'id':'positive','question':'已知?', 'expected_docs':['a.md'],'metadata':{'answerability':'answerable'}},
    ]
    records = collect_retrieval(samples, lambda texts:[[0.5]], Manager(), seed=42)
    assert len(records) == 2
    failure = next(r for r in records if r['mode'] == 'dense')
    assert failure['status'] == 'failed' and failure['docs'] is None
    assert failure['error_type'] == 'TimeoutError'
    success = next(r for r in records if r['mode'] == 'hybrid')
    assert success['docs'] == [] and success['status'] == 'success'
