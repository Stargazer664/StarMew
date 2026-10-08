"""文档向量化并写入 Milvus - 支持密集+稀疏向量"""
from .embedding import EmbeddingService, embedding_service as _default_embedding_service
from .milvus_client import MilvusManager
from .bm25 import lexical_text


class MilvusWriter:
    """文档向量化并写入 Milvus 服务 - 支持混合检索"""

    def __init__(self, embedding_service: EmbeddingService = None, milvus_manager: MilvusManager = None):
        self.embedding_service = embedding_service or _default_embedding_service
        self.milvus_manager = milvus_manager or MilvusManager()

    def prepare_dense(self, documents: list[dict], batch_size: int = 50) -> list[dict]:
        """先完成可能失败的稠密向量计算，不修改现有索引或 BM25。"""
        prepared = []
        for offset in range(0, len(documents), batch_size):
            batch = documents[offset:offset + batch_size]
            vectors = self.embedding_service.get_embeddings([doc["text"] for doc in batch])
            if len(vectors) != len(batch):
                raise ValueError("嵌入数量与文档分块数量不一致")
            for doc, vector in zip(batch, vectors):
                prepared.append({
                    "dense_embedding": vector, "text": doc["text"],
                    "filename": doc["filename"], "file_type": doc.get("file_type", ""),
                    "file_path": doc.get("file_path", ""), "page_number": doc.get("page_number", 0),
                    "chunk_idx": doc.get("chunk_idx", 0), "chunk_id": doc.get("chunk_id", ""),
                    "parent_chunk_id": doc.get("parent_chunk_id", ""),
                    "root_chunk_id": doc.get("root_chunk_id", ""), "chunk_level": doc.get("chunk_level", 0),
                })
        return prepared

    def write_prepared(self, prepared: list[dict], batch_size: int = 50, progress_callback=None):
        """写入已准备好的向量；异常由上层持久事务统一补偿。"""
        for offset in range(0, len(prepared), batch_size):
            batch = prepared[offset:offset + batch_size]
            if getattr(self.milvus_manager, "native_bm25", False):
                batch = [dict({k: v for k, v in row.items() if k != "sparse_embedding"},
                              lexical_text=lexical_text(row["text"])) for row in batch]
            self.milvus_manager.insert(batch)
            if progress_callback:
                progress_callback(min(offset + batch_size, len(prepared)), len(prepared))

    def write_documents(self, documents: list[dict], batch_size: int = 50, progress_callback=None):
        """
        批量写入文档到 Milvus（同时生成密集和稀疏向量）
        :param documents: 文档列表
        :param batch_size: 批次大小
        """
        if not documents:
            return

        self.milvus_manager.init_collection()

        if getattr(self.milvus_manager, "native_bm25", False):
            self.write_prepared(self.prepare_dense(documents, batch_size), batch_size, progress_callback)
            return

        all_texts = [doc["text"] for doc in documents]
        self.embedding_service.increment_add_documents(all_texts)

        total = len(documents)
        for i in range(0, total, batch_size):
            batch = documents[i:i + batch_size]
            texts = [doc["text"] for doc in batch]
            
            # 同时生成密集向量和稀疏向量
            dense_embeddings, sparse_embeddings = self.embedding_service.get_all_embeddings(texts)

            insert_data = [
                {
                    "dense_embedding": dense_emb,
                    "sparse_embedding": sparse_emb,
                    "text": doc["text"],
                    "filename": doc["filename"],
                    "file_type": doc["file_type"],
                    "file_path": doc.get("file_path", ""),
                    "page_number": doc.get("page_number", 0),
                    "chunk_idx": doc.get("chunk_idx", 0),
                    "chunk_id": doc.get("chunk_id", ""),
                    "parent_chunk_id": doc.get("parent_chunk_id", ""),
                    "root_chunk_id": doc.get("root_chunk_id", ""),
                    "chunk_level": doc.get("chunk_level", 0),
                }
                for doc, dense_emb, sparse_emb in zip(batch, dense_embeddings, sparse_embeddings)
            ]

            self.milvus_manager.insert(insert_data)

            # 每个批次写入后更新进度，前端据此展示“向量化入库 xx%”。
            if progress_callback:
                processed = min(i + batch_size, total)
                progress_callback(processed, total)
