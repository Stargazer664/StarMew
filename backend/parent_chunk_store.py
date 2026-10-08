"""父级分块文档存储（用于 Auto-merging Retriever）"""
from datetime import datetime
from typing import List

from .cache import cache
from .database import SessionLocal
from .models import ParentChunk


class ParentChunkStore:
    """基于 PostgreSQL + Redis 的父级分块存储。"""

    @staticmethod
    def _sanitize_text(value: str) -> str:
        """移除 PostgreSQL 不接受的 NUL (0x00) 字符。"""
        return value.replace("\x00", "") if value else value

    @staticmethod
    def _to_dict(item: ParentChunk) -> dict:
        return {
            "text": item.text,
            "filename": item.filename,
            "file_type": item.file_type,
            "file_path": item.file_path,
            "page_number": item.page_number,
            "chunk_id": item.chunk_id,
            "parent_chunk_id": item.parent_chunk_id,
            "root_chunk_id": item.root_chunk_id,
            "chunk_level": item.chunk_level,
            "chunk_idx": item.chunk_idx,
        }

    @staticmethod
    def _cache_key(chunk_id: str) -> str:
        return f"parent_chunk:{chunk_id}"

    def snapshot_documents(self, filename: str) -> list[dict]:
        with SessionLocal() as db:
            return [self._to_dict(row) for row in db.query(ParentChunk).filter(ParentChunk.filename == filename).all()]

    def replace_documents(self, filename: str, documents: list[dict]) -> None:
        """在一个 PostgreSQL 事务内替换父块，提交后才清理缓存。"""
        with SessionLocal() as db:
            old_ids = [row.chunk_id for row in db.query(ParentChunk).filter(ParentChunk.filename == filename).all()]
            new_ids = []
            try:
                db.query(ParentChunk).filter(ParentChunk.filename == filename).delete(synchronize_session=False)
                for doc in documents:
                    if doc.get("filename") != filename or not doc.get("chunk_id"):
                        raise ValueError("父块文件名或 chunk_id 无效")
                    new_ids.append(doc["chunk_id"])
                    payload = {key: doc.get(key, "") for key in (
                        "chunk_id", "filename", "text", "file_type", "file_path", "parent_chunk_id", "root_chunk_id",
                    )}
                    payload = {key: self._sanitize_text(value) for key, value in payload.items()}
                    payload.update({key: int(doc.get(key, 0) or 0) for key in ("page_number", "chunk_level", "chunk_idx")})
                    db.add(ParentChunk(**payload))
                db.commit()
            except Exception:
                db.rollback()
                raise
        for chunk_id in set(old_ids + new_ids):
            cache.delete(self._cache_key(chunk_id))

    def upsert_documents(self, docs: List[dict]) -> int:
        """写入/更新父级分块，返回写入条数。"""
        if not docs:
            return 0

        db = SessionLocal()
        upserted = 0
        try:
            for doc in docs:
                chunk_id = (doc.get("chunk_id") or "").strip()
                if not chunk_id:
                    continue

                record = db.query(ParentChunk).filter(ParentChunk.chunk_id == chunk_id).first()
                payload = {
                    "text": self._sanitize_text(doc.get("text", "")),
                    "filename": self._sanitize_text(doc.get("filename", "")),
                    "file_type": self._sanitize_text(doc.get("file_type", "")),
                    "file_path": self._sanitize_text(doc.get("file_path", "")),
                    "page_number": int(doc.get("page_number", 0) or 0),
                    "parent_chunk_id": self._sanitize_text(doc.get("parent_chunk_id", "")),
                    "root_chunk_id": self._sanitize_text(doc.get("root_chunk_id", "")),
                    "chunk_level": int(doc.get("chunk_level", 0) or 0),
                    "chunk_idx": int(doc.get("chunk_idx", 0) or 0),
                    "updated_at": datetime.utcnow(),
                }
                cache_payload = {
                    "chunk_id": chunk_id,
                    "text": payload["text"],
                    "filename": payload["filename"],
                    "file_type": payload["file_type"],
                    "file_path": payload["file_path"],
                    "page_number": payload["page_number"],
                    "parent_chunk_id": payload["parent_chunk_id"],
                    "root_chunk_id": payload["root_chunk_id"],
                    "chunk_level": payload["chunk_level"],
                    "chunk_idx": payload["chunk_idx"],
                }
                if record:
                    for key, value in payload.items():
                        setattr(record, key, value)
                else:
                    db.add(ParentChunk(chunk_id=chunk_id, **payload))

                cache.set_json(self._cache_key(chunk_id), cache_payload)
                upserted += 1

            db.commit()
        finally:
            db.close()

        return upserted

    def get_documents_by_ids(self, chunk_ids: List[str]) -> List[dict]:
        """父块 ID 会在替换时复用，直接读数据库以避开失效失败的旧缓存。"""
        keys = [(chunk_id or "").strip() for chunk_id in chunk_ids]
        keys = [key for key in keys if key]
        if not keys:
            return []

        with SessionLocal() as db:
            rows = db.query(ParentChunk).filter(ParentChunk.chunk_id.in_(keys)).all()
            ordered_results = {row.chunk_id: self._to_dict(row) for row in rows}

        return [ordered_results[item] for item in chunk_ids if item in ordered_results]

    def delete_by_filename(self, filename: str) -> int:
        """按文件名删除父级分块，返回删除条数。"""
        if not filename:
            return 0

        db = SessionLocal()
        try:
            rows = db.query(ParentChunk).filter(ParentChunk.filename == filename).all()
            chunk_ids = [row.chunk_id for row in rows]
            deleted = len(chunk_ids)
            if deleted > 0:
                db.query(ParentChunk).filter(ParentChunk.filename == filename).delete(synchronize_session=False)
                db.commit()
                for chunk_id in chunk_ids:
                    cache.delete(self._cache_key(chunk_id))
            return deleted
        finally:
            db.close()
