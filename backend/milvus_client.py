"""Milvus 客户端 - 支持密集向量+稀疏向量混合检索"""
import os
import threading
from typing import Callable, TypeVar

from dotenv import load_dotenv
from pymilvus import MilvusClient, DataType, AnnSearchRequest, RRFRanker, Function, FunctionType
from .agent_budget import network_timeout_kwargs
from .bm25 import BM25_SCHEMA_VERSION, BM25MigrationRequired, lexical_text, validate_schema

load_dotenv()

# Milvus 单次 query 的 limit 上限（超出会报 invalid max query result window）
QUERY_MAX_LIMIT = 16384
T = TypeVar("T")


class MilvusManager:
    """Milvus 连接和集合管理 - 支持混合检索"""
    native_bm25 = True

    def __init__(self):
        self.host = os.getenv("MILVUS_HOST", "localhost")
        self.port = os.getenv("MILVUS_PORT", "19530")
        self.collection_name = os.getenv("MILVUS_COLLECTION", "embeddings_collection")
        self.uri = f"http://{self.host}:{self.port}"
        self.client = None
        self._client_lock = threading.RLock()
        self._bm25_validated = False

    def _get_client(self) -> MilvusClient:
        # Lazy-create client to avoid blocking app import/startup when Milvus is temporarily unavailable.
        with self._client_lock:
            if self.client is None:
                self.client = MilvusClient(uri=self.uri, **network_timeout_kwargs())
            return self.client

    @staticmethod
    def _is_closed_channel_error(exc: Exception) -> bool:
        return "closed channel" in str(exc).lower()

    @staticmethod
    def _close_client(client) -> None:
        close = getattr(client, "close", None)
        if not callable(close):
            return
        try:
            close()
        except Exception:
            pass

    def _reset_client(self, failed_client=None) -> None:
        with self._client_lock:
            if self.client is None:
                return
            if failed_client is not None and self.client is not failed_client:
                return
            client = self.client
            self.client = None
            self._bm25_validated = False

        self._close_client(client)

    def _run_with_reconnect(self, operation: Callable[[MilvusClient], T]) -> T:
        client = self._get_client()
        try:
            return operation(client)
        except Exception as exc:
            if not self._is_closed_channel_error(exc):
                raise

            self._reset_client(client)
            return operation(self._get_client())

    def init_collection(self, dense_dim: int | None = None):
        """
        初始化 Milvus 集合 - 同时支持密集向量和稀疏向量
        :param dense_dim: 密集向量维度；默认读环境变量 DENSE_EMBEDDING_DIM（本地 BAAI/bge-m3 为 1024）
        """
        if dense_dim is None:
            dense_dim = int(os.getenv("DENSE_EMBEDDING_DIM", "1024"))
        def _init(client: MilvusClient) -> None:
            if not client.has_collection(self.collection_name):
                schema = client.create_schema(auto_id=True, enable_dynamic_field=True, description=BM25_SCHEMA_VERSION)

                # 主键
                schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
                
                # 密集向量（来自 embedding 模型）
                schema.add_field("dense_embedding", DataType.FLOAT_VECTOR, dim=dense_dim)
                
                # 稀疏向量（来自 BM25）
                schema.add_field("sparse_embedding", DataType.SPARSE_FLOAT_VECTOR)
                
                # 文本和元数据字段
                schema.add_field("text", DataType.VARCHAR, max_length=2000)
                schema.add_field("lexical_text", DataType.VARCHAR, max_length=65535,
                                 enable_analyzer=True, analyzer_params={"tokenizer": "whitespace"})
                schema.add_function(Function(name="native_bm25", input_field_names=["lexical_text"],
                    output_field_names=["sparse_embedding"], function_type=FunctionType.BM25))
                schema.add_field("filename", DataType.VARCHAR, max_length=255)
                schema.add_field("file_type", DataType.VARCHAR, max_length=50)
                schema.add_field("file_path", DataType.VARCHAR, max_length=1024)
                schema.add_field("page_number", DataType.INT64)
                schema.add_field("chunk_idx", DataType.INT64)

                # Auto-merging 所需层级字段
                schema.add_field("chunk_id", DataType.VARCHAR, max_length=512)
                schema.add_field("parent_chunk_id", DataType.VARCHAR, max_length=512)
                schema.add_field("root_chunk_id", DataType.VARCHAR, max_length=512)
                schema.add_field("chunk_level", DataType.INT64)

                # 为两种向量分别创建索引
                index_params = client.prepare_index_params()
                
                # 密集向量索引 - 使用 HNSW（更适合混合检索）
                index_params.add_index(
                    field_name="dense_embedding",
                    index_type="HNSW",
                    metric_type="IP",
                    params={"M": 16, "efConstruction": 256}
                )
                
                # 稀疏向量索引
                index_params.add_index(
                    field_name="sparse_embedding",
                    index_type="SPARSE_INVERTED_INDEX",
                    metric_type="BM25",
                    params={"inverted_index_algo": "TAAT_NAIVE", "bm25_k1": 1.5, "bm25_b": 0.75}
                )

                client.create_collection(
                    collection_name=self.collection_name,
                    schema=schema,
                    index_params=index_params
                )
                self._bm25_validated = True
            else:
                self.assert_bm25_compatible()

        self._run_with_reconnect(_init)

    def assert_bm25_compatible(self):
        """Never run the new query encoder against legacy IP vectors."""
        if self._bm25_validated:
            return
        client = self._get_client()
        validate_schema(client.describe_collection(self.collection_name, **network_timeout_kwargs()))
        index = client.describe_index(self.collection_name, "sparse_embedding", **network_timeout_kwargs())
        if index.get("metric_type") != "BM25" or index.get("inverted_index_algo") != "TAAT_NAIVE":
            raise BM25MigrationRequired("BM25 索引参数不匹配，请重建为 BM25/TAAT_NAIVE")
        self._bm25_validated = True

    def insert(self, data: list[dict]):
        """插入不能在响应不明时自动重放（auto_id 会产生重复行）。"""
        client = self._get_client()
        try:
            # PyMilvus 自身也会重试 RPC；仅关闭本类的重连重放还不够。
            # SDK 指定 timeout 时按时间预算重试，会覆盖 retry_times 的限制。
            return client.insert(self.collection_name, data, retry_times=0, timeout=None)
        except Exception as exc:
            if self._is_closed_channel_error(exc):
                self._reset_client(client)
            raise

    def snapshot_documents(self, filename: str) -> list[dict]:
        """使用强一致游标读取恢复快照，不受 offset 查询窗口限制。"""
        import json
        fields = ["dense_embedding", "lexical_text", "text", "filename", "file_type", "file_path",
                  "page_number", "chunk_idx", "chunk_id", "parent_chunk_id", "root_chunk_id", "chunk_level"]
        iterator = self._get_client().query_iterator(
            collection_name=self.collection_name, batch_size=1000,
            filter=f"filename == {json.dumps(filename, ensure_ascii=False)}",
            output_fields=fields, consistency_level="Strong",
        )
        rows = []
        try:
            while True:
                batch = iterator.next()
                if not batch:
                    return rows
                rows.extend({key: row[key] for key in fields if key in row} for row in batch)
        finally:
            iterator.close()

    def query(
        self,
        filter_expr: str = "",
        output_fields: list[str] = None,
        limit: int = 10000,
        offset: int = 0,
    ):
        """查询数据。limit 不宜超过 QUERY_MAX_LIMIT。"""
        return self._run_with_reconnect(
            lambda client: client.query(
                collection_name=self.collection_name,
                filter=filter_expr,
                output_fields=output_fields or ["filename", "file_type"],
                limit=min(limit, QUERY_MAX_LIMIT),
                offset=offset,
                consistency_level="Strong",
            )
        )

    def query_all(self, filter_expr: str = "", output_fields: list[str] | None = None) -> list:
        """分页拉取匹配 filter 的全部行，避免单次 limit 超过服务端窗口。"""
        fields = output_fields or ["filename", "file_type"]
        out: list = []
        offset = 0
        while True:
            batch = self._run_with_reconnect(
                lambda client: client.query(
                    collection_name=self.collection_name,
                    filter=filter_expr,
                    output_fields=fields,
                    limit=QUERY_MAX_LIMIT,
                    offset=offset,
                    consistency_level="Strong",
                )
            )
            if not batch:
                break
            out.extend(batch)
            if len(batch) < QUERY_MAX_LIMIT:
                break
            offset += len(batch)
        return out

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[dict]:
        """根据 chunk_id 批量查询分块（用于 Auto-merging 拉取父块）"""
        ids = [item for item in chunk_ids if item]
        if not ids:
            return []
        quoted_ids = ", ".join([f'"{item}"' for item in ids])
        filter_expr = f"chunk_id in [{quoted_ids}]"
        return self.query(
            filter_expr=filter_expr,
            output_fields=[
                "text",
                "filename",
                "file_type",
                "page_number",
                "chunk_id",
                "parent_chunk_id",
                "root_chunk_id",
                "chunk_level",
                "chunk_idx",
            ],
            limit=len(ids),
        )

    def hybrid_retrieve(
        self,
        dense_embedding: list[float],
        query_text: str,
        top_k: int = 5,
        rrf_k: int = 60,     #可调节
        filter_expr: str = "",
    ) -> list[dict]:
        """
        混合检索 - 使用 RRF 融合密集向量和稀疏向量的检索结果
        
        :param dense_embedding: 密集向量
        :param query_text: 原始查询文本，由原生 BM25 在服务端统一评分
        :param top_k: 返回结果数量
        :param rrf_k: RRF 算法参数 k，默认60
        :return: 检索结果列表
        """
        self.assert_bm25_compatible()
        sparse_text = lexical_text(query_text)
        if not sparse_text:
            return self.dense_retrieve(dense_embedding, top_k, filter_expr)
        output_fields = [
            "text",
            "filename",
            "file_type",
            "page_number",
            "chunk_id",
            "parent_chunk_id",
            "root_chunk_id",
            "chunk_level",
            "chunk_idx",
        ]
        
        # 密集向量搜索请求
        dense_search = AnnSearchRequest(
            data=[dense_embedding],
            anns_field="dense_embedding",
            param={"metric_type": "IP", "params": {"ef": 64}},
            limit=top_k * 2,  # 多取一些用于融合
            expr=filter_expr,
        )
        
        # 稀疏向量搜索请求
        sparse_search = AnnSearchRequest(
            data=[sparse_text],
            anns_field="sparse_embedding",
            param={"metric_type": "BM25", "params": {}},
            limit=top_k * 2,
            expr=filter_expr,
        )
        
        # 使用 RRF 排序算法融合结果
        reranker = RRFRanker(k=rrf_k)
        
        results = self._run_with_reconnect(
            lambda client: client.hybrid_search(
                collection_name=self.collection_name,
                reqs=[dense_search, sparse_search],
                ranker=reranker,
                limit=top_k,
                output_fields=output_fields,
                consistency_level="Strong",
                **network_timeout_kwargs(),
            )
        )
        
        # 格式化返回结果
        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append({
                    "id": hit.get("id"),
                    "text": hit.get("text", ""),
                    "filename": hit.get("filename", ""),
                    "file_type": hit.get("file_type", ""),
                    "page_number": hit.get("page_number", 0),
                    "chunk_id": hit.get("chunk_id", ""),
                    "parent_chunk_id": hit.get("parent_chunk_id", ""),
                    "root_chunk_id": hit.get("root_chunk_id", ""),
                    "chunk_level": hit.get("chunk_level", 0),
                    "chunk_idx": hit.get("chunk_idx", 0),
                    "score": hit.get("distance", 0.0)
                })
        
        return formatted_results

    def dense_retrieve(self, dense_embedding: list[float], top_k: int = 5, filter_expr: str = "") -> list[dict]:
        """
        仅使用密集向量检索（降级模式，用于稀疏向量不可用时）
        """
        results = self._run_with_reconnect(
            lambda client: client.search(
                collection_name=self.collection_name,
                data=[dense_embedding],
                anns_field="dense_embedding",
                search_params={"metric_type": "IP", "params": {"ef": 64}},
                limit=top_k,
                output_fields=[
                    "text",
                    "filename",
                    "file_type",
                    "page_number",
                    "chunk_id",
                    "parent_chunk_id",
                    "root_chunk_id",
                    "chunk_level",
                    "chunk_idx",
                ],
                filter=filter_expr,
                consistency_level="Strong",
                **network_timeout_kwargs(),
            )
        )
        
        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append({
                    "id": hit.get("id"),
                    "text": hit.get("entity", {}).get("text", ""),
                    "filename": hit.get("entity", {}).get("filename", ""),
                    "file_type": hit.get("entity", {}).get("file_type", ""),
                    "page_number": hit.get("entity", {}).get("page_number", 0),
                    "chunk_id": hit.get("entity", {}).get("chunk_id", ""),
                    "parent_chunk_id": hit.get("entity", {}).get("parent_chunk_id", ""),
                    "root_chunk_id": hit.get("entity", {}).get("root_chunk_id", ""),
                    "chunk_level": hit.get("entity", {}).get("chunk_level", 0),
                    "chunk_idx": hit.get("entity", {}).get("chunk_idx", 0),
                    "score": hit.get("distance", 0.0)
                })
        
        return formatted_results

    def delete(self, filter_expr: str):
        """删除数据"""
        return self._run_with_reconnect(
            lambda client: client.delete(
                collection_name=self.collection_name,
                filter=filter_expr
            )
        )

    def has_collection(self) -> bool:
        """检查集合是否存在"""
        return self._run_with_reconnect(lambda client: client.has_collection(self.collection_name))

    def drop_collection(self):
        """删除集合（用于重建 schema）"""
        def _drop(client: MilvusClient) -> None:
            if client.has_collection(self.collection_name):
                client.drop_collection(self.collection_name)

        self._run_with_reconnect(_drop)
