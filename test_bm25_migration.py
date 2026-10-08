import copy
from types import SimpleNamespace

import pytest
from pymilvus import MilvusClient

from backend.bm25 import lexical_text
from backend.bm25_migration import FIELDS, migrate, activate


class Database:
    create_schema = staticmethod(MilvusClient.create_schema)

    def __init__(self):
        self.rows = {"old": [dict(text="GPT-4 北京", dense_embedding=[1.0, 0.0],
            filename="a.md", file_type="Markdown", file_path="a.md", page_number=0,
            chunk_idx=0, chunk_id="leaf", parent_chunk_id="parent", root_chunk_id="root", chunk_level=3)]}
        self.fail_insert = False
        self.corrupt = False
        self.closed = 0

    def has_collection(self, name):
        return name in self.rows

    def describe_collection(self, name):
        return {"enable_dynamic_field": True,
            "fields": [{"name": k, "params": {"dim": 2} if k == "dense_embedding" else {}}
            for k in FIELDS]}

    def prepare_index_params(self):
        return SimpleNamespace(add_index=lambda **kw: None)

    def create_collection(self, collection_name, **kwargs):
        self.rows[collection_name] = []

    def query_iterator(self, collection_name, **kwargs):
        assert kwargs["consistency_level"] == "Strong"
        assert "sparse_embedding" not in kwargs["output_fields"]
        rows = [{k: copy.deepcopy(v) for k, v in row.items()
                 if k in kwargs["output_fields"] or (k not in FIELDS and "$meta" in kwargs["output_fields"])}
                for row in self.rows[collection_name]]
        if self.corrupt and collection_name != "old":
            rows[0]["text"] = "corrupted"
        batches = iter([rows, []])
        def close():
            self.closed += 1
        return SimpleNamespace(next=lambda: next(batches), close=close)

    def insert(self, name, data, **kwargs):
        assert kwargs["retry_times"] == 0 and kwargs["timeout"] is None
        self.rows[name].extend(copy.deepcopy(data))
        if self.fail_insert:
            raise RuntimeError("lost insert acknowledgement")

    def flush(self, name):
        pass


def test_migration_copies_dense_and_metadata_verifies_before_activation(tmp_path):
    db = Database()
    before = copy.deepcopy(db.rows["old"])
    report = migrate(db, "old", "new", batch_size=1)
    assert report["verified"] and report["rows"] == 1
    assert db.rows["old"] == before
    assert db.rows["new"] == [dict(before[0], lexical_text=lexical_text(before[0]["text"]))]
    assert db.closed == 3  # source snapshot, target verification, source recheck
    env = tmp_path / ".env"
    env.write_text("# configuration\nMILVUS_COLLECTION=old\nSECRET=keep\n", encoding="utf-8")
    original = env.read_bytes()
    backup = activate(report, env)
    assert backup.read_bytes() == original
    assert "MILVUS_COLLECTION=new" in env.read_text(encoding="utf-8")
    assert "SECRET=keep" in env.read_text(encoding="utf-8")


@pytest.mark.parametrize("failure", ["fail_insert", "corrupt"])
def test_failure_preserves_source_and_never_switches_config(tmp_path, failure):
    db = Database()
    setattr(db, failure, True)
    before = copy.deepcopy(db.rows["old"])
    env = tmp_path / ".env"
    env.write_text("MILVUS_COLLECTION=old\n", encoding="utf-8")
    with pytest.raises(RuntimeError):
        report = migrate(db, "old", "new")
        activate(report, env)
    assert db.rows["old"] == before
    assert env.read_text(encoding="utf-8") == "MILVUS_COLLECTION=old\n"
    assert "new" in db.rows


def test_refuses_existing_target_and_unverified_activation(tmp_path):
    db = Database()
    with pytest.raises(ValueError):
        migrate(db, "old", "old")
    db.rows["new"] = []
    with pytest.raises(ValueError):
        migrate(db, "old", "new")
    with pytest.raises(ValueError):
        activate({"verified": False}, tmp_path / ".env")


def test_activation_refuses_unrelated_config(tmp_path):
    env = tmp_path / ".env"
    env.write_text("MILVUS_COLLECTION=another\n", encoding="utf-8")
    with pytest.raises(ValueError):
        activate({"verified": True, "source": "old", "target": "new"}, env)
    assert env.read_text(encoding="utf-8") == "MILVUS_COLLECTION=another\n"


def test_dynamic_metadata_survives_migration_and_is_verified():
    db = Database()
    db.rows["old"][0]["source_tag"] = {"author": "test", "version": 2}
    report = migrate(db, "old", "new")
    assert report["verified"]
    assert db.rows["new"][0]["source_tag"] == db.rows["old"][0]["source_tag"]


def test_dynamic_metadata_corruption_prevents_verification():
    db = Database()
    db.rows["old"][0]["source_tag"] = "important"
    insert = db.insert
    def corrupt(name, data, **kwargs):
        insert(name, data, **kwargs)
        db.rows[name][0]["source_tag"] = "wrong"
    db.insert = corrupt
    with pytest.raises(RuntimeError, match="核对失败"):
        migrate(db, "old", "new")
