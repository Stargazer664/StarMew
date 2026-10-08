import copy
import json
from types import SimpleNamespace

import pytest

from backend.embedding import EmbeddingService
from backend.milvus_writer import MilvusWriter


class ProcessCrash(BaseException):
    pass


class VectorStore:
    def __init__(self, rows):
        self.rows = copy.deepcopy(rows)
        self.fail_insert = False
        self.fail_delete = False
        self.offline = False
        self.crash_insert = False

    def init_collection(self):
        pass

    def snapshot_documents(self, filename):
        return copy.deepcopy([r for r in self.rows if r['filename'] == filename])

    def delete(self, expr):
        if self.offline:
            raise RuntimeError('storage offline')
        if self.fail_delete:
            self.fail_delete = False
            raise RuntimeError('delete failed')
        filename = json.loads(expr.split(' == ', 1)[1])
        self.rows = [r for r in self.rows if r['filename'] != filename]

    def insert(self, rows):
        if self.offline:
            raise RuntimeError('storage offline')
        self.rows.extend(copy.deepcopy(rows))
        if self.crash_insert:
            self.crash_insert = False
            raise ProcessCrash()
        if self.fail_insert:
            self.fail_insert = False
            raise RuntimeError('insert acknowledgement lost')


class ParentStore:
    def __init__(self, rows):
        self.rows = copy.deepcopy(rows)
        self.fail_replace = False

    def snapshot_documents(self, filename):
        return copy.deepcopy([r for r in self.rows if r['filename'] == filename])

    def replace_documents(self, filename, docs):
        if self.fail_replace:
            self.fail_replace = False
            raise RuntimeError('parent write failed')
        self.rows = [r for r in self.rows if r['filename'] != filename] + copy.deepcopy(docs)


def make_service(tmp_path):
    from backend.document_transactions import DocumentTransactionService
    upload_root = tmp_path / 'documents'
    upload_root.mkdir(exist_ok=True)
    original = upload_root / 'guide.md'
    original.write_text('old file', encoding='utf-8')
    staging = upload_root / '.staging' / ('a' * 32) / 'guide.md'
    staging.parent.mkdir(parents=True, exist_ok=True)
    staging.write_text('new file', encoding='utf-8')
    old_parent = dict(filename='guide.md', text='old parent', chunk_id='old-parent', chunk_level=2)
    old_vector = dict(filename='guide.md', text='old leaf', chunk_id='old-leaf',
                      chunk_level=3, dense_embedding=[0.5, 0.5], sparse_embedding={0: 0.3})
    embedding = EmbeddingService(tmp_path / 'bm25.json')
    embedding.increment_add_documents(['old leaf'])
    embedding._embedder = SimpleNamespace(embed_documents=lambda texts: [[1.0, 0.0] for _ in texts])
    embedding._embedder_ready = True
    vectors, parents = VectorStore([old_vector]), ParentStore([old_parent])
    docs = [dict(filename='guide.md', text='new parent', chunk_id='new-parent', chunk_level=2),
            dict(filename='guide.md', text='new leaf', chunk_id='new-leaf', chunk_level=3, file_type='Markdown')]
    loader = SimpleNamespace(load_document=lambda *args: copy.deepcopy(docs))
    writer = MilvusWriter(embedding, vectors)
    service = DocumentTransactionService(tmp_path / 'transactions', upload_root, loader, writer, parents, embedding)
    return service, staging, original, vectors, parents, embedding


@pytest.mark.parametrize('failure', ['embedding', 'delete', 'parent', 'insert', 'file'])
def test_failed_replacement_restores_old_file_parents_vectors_and_bm25(tmp_path, monkeypatch, failure):
    service, staging, original, vectors, parents, embedding = make_service(tmp_path)
    before = (copy.deepcopy(vectors.rows), copy.deepcopy(parents.rows), embedding.snapshot_state())
    if failure == 'embedding':
        embedding._embedder.embed_documents = lambda _: (_ for _ in ()).throw(RuntimeError('embedding down'))
    elif failure == 'delete':
        vectors.fail_delete = True
    elif failure == 'parent':
        parents.fail_replace = True
    elif failure == 'insert':
        vectors.fail_insert = True
    else:
        def failed_publish(*args):
            original.write_text('partially published', encoding='utf-8')
            raise OSError('file publish failed')
        monkeypatch.setattr(service, '_publish_file', failed_publish)
    with pytest.raises(Exception):
        service.replace('a' * 32, staging, 'guide.md')
    assert original.read_text(encoding='utf-8') == 'old file'
    assert staging.read_text(encoding='utf-8') == 'new file'
    assert vectors.rows == before[0]
    assert parents.rows == before[1]
    assert embedding.snapshot_state() == before[2]


def test_success_publishes_once_and_updates_statistics(tmp_path):
    service, staging, original, vectors, parents, embedding = make_service(tmp_path)
    result = service.replace('a' * 32, staging, 'guide.md')
    assert result == {'parent_count': 1, 'leaf_count': 1}
    assert original.read_text(encoding='utf-8') == 'new file'
    assert [row['text'] for row in vectors.rows] == ['new leaf']
    assert [row['text'] for row in parents.rows] == ['new parent']
    state = embedding.snapshot_state()
    assert state['total_docs'] == 1
    assert 'old' not in state['doc_freq'] and state['doc_freq']['new'] == 1
    recovered = service.recover()
    assert recovered['a' * 32]['status'] == 'completed'
    assert vectors.rows[0]['text'] == 'new leaf'
    service.acknowledge('a' * 32)
    assert service.recover() == {}


def test_restart_rolls_back_interrupted_commit_idempotently(tmp_path):
    service, staging, original, vectors, parents, embedding = make_service(tmp_path)
    before = (copy.deepcopy(vectors.rows), copy.deepcopy(parents.rows), embedding.snapshot_state())
    vectors.crash_insert = True
    with pytest.raises(ProcessCrash):
        service.replace('a' * 32, staging, 'guide.md')
    # New instance, same persisted journal and existing stores: no process memory is used.
    restarted = type(service)(service.root, service.upload_root, service.loader, service.writer, parents, embedding)
    assert restarted.recover()['a' * 32]['status'] == 'failed'
    assert restarted.recover()['a' * 32]['status'] == 'failed'
    assert (vectors.rows, parents.rows, embedding.snapshot_state()) == before
    assert original.read_text(encoding='utf-8') == 'old file'


def test_unfinished_rollback_blocks_next_write_until_recovery(tmp_path, monkeypatch):
    service, staging, original, vectors, parents, embedding = make_service(tmp_path)
    before = copy.deepcopy(vectors.rows)
    real_insert = vectors.insert
    def lose_storage(rows):
        real_insert(rows)
        vectors.offline = True
        raise RuntimeError('storage lost during write')
    monkeypatch.setattr(vectors, 'insert', lose_storage)
    with pytest.raises(Exception, match='恢复'):
        service.replace('a' * 32, staging, 'guide.md')
    with pytest.raises(Exception, match='恢复'):
        service.replace('b' * 32, staging, 'guide.md')
    assert list(service.root.glob('*.json'))
    vectors.offline = False
    monkeypatch.setattr(vectors, 'insert', real_insert)
    assert service.recover()['a' * 32]['status'] == 'failed'
    assert vectors.rows == before


def test_first_upload_failure_removes_partial_new_data(tmp_path):
    service, staging, original, vectors, parents, embedding = make_service(tmp_path)
    original.unlink()
    vectors.rows.clear()
    parents.rows.clear()
    embedding.increment_remove_documents(['old leaf'])
    before = embedding.snapshot_state()
    vectors.fail_insert = True
    with pytest.raises(Exception):
        service.replace('a' * 32, staging, 'guide.md')
    assert not original.exists()
    assert vectors.rows == parents.rows == []
    assert embedding.snapshot_state() == before


def test_snapshot_persistence_failure_never_removes_old_data(tmp_path, monkeypatch):
    service, staging, original, vectors, parents, embedding = make_service(tmp_path)
    before = copy.deepcopy(vectors.rows)
    monkeypatch.setattr(service, '_save_manifest', lambda *_: (_ for _ in ()).throw(OSError('disk full')))
    with pytest.raises(OSError):
        service.replace('a' * 32, staging, 'guide.md')
    assert vectors.rows == before
    assert original.read_text(encoding='utf-8') == 'old file'
