"""Non-destructive native BM25 migration; caller must hold backend ownership.

Stream rows through cursors and compare multisets of SHA256 row digests. The
source and any partially populated destination are deliberately never deleted.
"""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import uuid

from dotenv import dotenv_values

from .bm25 import BM25_SCHEMA_VERSION, lexical_text
from .milvus_client import MilvusManager

FIELDS = ["dense_embedding", "text", "filename", "file_type", "file_path",
          "page_number", "chunk_idx", "chunk_id", "parent_chunk_id", "root_chunk_id", "chunk_level"]


def _name(value):
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,254}", value):
        raise ValueError("Milvus 集合名无效")
    return value


def _batches(client, name, fields, batch_size, *, dynamic=False):
    iterator = client.query_iterator(collection_name=name, batch_size=batch_size,
        filter="", output_fields=fields + (["$meta"] if dynamic else []), consistency_level="Strong")
    try:
        while batch := iterator.next():
            # PyMilvus flattens requested $meta into the row. Preserve all such
            # keys, while dropping the auto-generated primary key only.
            yield [{key: value for key, value in row.items() if key != "id"}
                   if dynamic else {key: row[key] for key in fields} for row in batch]
    finally:
        iterator.close()


def _digest(row):
    data = json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":"), default=lambda x: x.tolist())
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def migrate(client, source, target, *, batch_size=500, progress=lambda message: None):
    _name(source)
    _name(target)
    if source == target or client.has_collection(target):
        raise ValueError("目标必须是尚不存在的新集合；旧集合和失败的目标集合不会被覆盖")
    if batch_size <= 0 or not client.has_collection(source):
        raise ValueError("源集合不存在或 batch_size 无效")
    description = client.describe_collection(source)
    source_dynamic = description.get("enable_dynamic_field", False)
    fields = {f["name"]: f for f in description["fields"]}
    if not set(FIELDS).issubset(fields):
        raise ValueError("源集合缺少迁移所需字段，请先检查旧版本 schema")
    unknown = set(fields) - set(FIELDS) - {"id", "sparse_embedding", "lexical_text", "$meta"}
    if unknown:
        raise ValueError("源集合含有未支持字段，停止迁移以免丢失数据：" + ", ".join(sorted(unknown)))
    destination = MilvusManager()
    destination.client, destination.collection_name = client, target
    destination.init_collection(dense_dim=int(fields["dense_embedding"]["params"]["dim"]))
    expected, count = Counter(), 0
    for batch in _batches(client, source, FIELDS, batch_size, dynamic=source_dynamic):
        prepared = [dict(row, lexical_text=lexical_text(row["text"])) for row in batch]
        destination.insert(prepared)  # Never replay an ambiguous auto-id insert.
        expected.update(_digest(row) for row in prepared)
        count += len(prepared)
        progress(f"已复制 {count} 个分块")
    client.flush(target)
    actual = Counter()
    for batch in _batches(client, target, FIELDS + ["lexical_text"], batch_size, dynamic=True):
        actual.update(_digest(row) for row in batch)
    if actual != expected:
        raise RuntimeError("新集合全文、向量或元数据核对失败；未切换配置，请保留现场检查")
    # Detect source changes by other clients during the copy. The local backend
    # lock excludes normal application writes; remote writers must be stopped.
    current = Counter()
    for batch in _batches(client, source, FIELDS, batch_size, dynamic=source_dynamic):
        current.update(_digest(dict(row, lexical_text=lexical_text(row["text"]))) for row in batch)
    if current != expected:
        raise RuntimeError("迁移期间源集合发生变化；未切换配置")
    progress(f"核对通过：{count} 个分块")
    return {"source": source, "target": target, "rows": count, "verified": True,
            "schema_version": BM25_SCHEMA_VERSION}


def activate(report, env_file):
    """Atomically select a verified collection and keep the exact old .env bytes."""
    if report.get("verified") is not True:
        raise ValueError("仅允许启用已经完整核对的新集合")
    source, target = _name(report["source"]), _name(report["target"])
    path = Path(env_file).resolve()
    original = path.read_bytes()  # Explicitly require a real deployment config.
    configured = dotenv_values(path).get("MILVUS_COLLECTION") or "embeddings_collection"
    if configured != source:
        raise ValueError(".env 当前集合与迁移源不一致，停止自动切换")
    text = original.decode("utf-8-sig")
    pattern = r"(?m)^[ \t]*(?:export[ \t]+)?MILVUS_COLLECTION[ \t]*=.*$"
    updated, matches = re.subn(pattern, f"MILVUS_COLLECTION={target}", text)
    if not matches:
        updated = text.rstrip("\r\n") + f"\nMILVUS_COLLECTION={target}\n"
    backup = path.with_name(path.name + ".bm25." + uuid.uuid4().hex + ".bak")
    with backup.open("xb") as handle:
        handle.write(original)
        handle.flush()
        os.fsync(handle.fileno())
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".bm25.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(updated.encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        if path.read_bytes() != original:
            raise RuntimeError("迁移配置被并发修改，停止切换")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return backup
