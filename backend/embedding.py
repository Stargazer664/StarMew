"""本地稠密向量服务；旧 BM25 统计/批量编码仅保留用于旧格式兼容。"""
import json
import math
import os
import re
import threading
from copy import deepcopy
from collections import Counter
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from .telemetry import span, timed

# 延迟导入 HuggingFaceEmbeddings
_HuggingFaceEmbeddings = None

def _get_embedder():
    global _HuggingFaceEmbeddings
    if _HuggingFaceEmbeddings is None:
        from langchain_huggingface import HuggingFaceEmbeddings
        _HuggingFaceEmbeddings = HuggingFaceEmbeddings
    return _HuggingFaceEmbeddings

load_dotenv()

_DEFAULT_STATE_PATH = Path(__file__).resolve().parent.parent / "data" / "bm25_state.json"


def _create_dense_embedder():
    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    device = os.getenv("EMBEDDING_DEVICE", "cpu")
    # Normalize non-standard device names (e.g., "gpu" → "cuda")
    _device_lower = device.strip().lower()
    if _device_lower in ("gpu", "cuda", "0"):
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    HuggingFaceEmbeddings = _get_embedder()
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


class EmbeddingService:
    """本地稠密嵌入，以及旧格式 BM25 状态恢复接口。"""

    def __init__(self, state_path: Path | str | None = None):
        self._embedder = None  # 延迟加载
        self._embedder_ready = False
        self._state_path = Path(state_path or os.getenv("BM25_STATE_PATH", _DEFAULT_STATE_PATH))
        self._lock = threading.RLock()

        # BM25 参数
        self.k1 = 1.5
        self.b = 0.75

        self._vocab: dict[str, int] = {}
        self._vocab_counter = 0
        self._doc_freq: Counter[str] = Counter()
        self._total_docs = 0
        self._sum_token_len = 0
        self._avg_doc_len = 1.0

        # Native BM25 does not depend on a local vocabulary. Explicit paths are
        # retained for legacy transaction recovery and offline tooling only.
        if state_path is not None:
            self._load_state()

    def _recompute_avg_len(self) -> None:
        self._avg_doc_len = (
            self._sum_token_len / self._total_docs if self._total_docs > 0 else 1.0
        )

    def _load_state(self, *, strict: bool = False) -> None:
        path = self._state_path
        if not path.is_file():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            if strict:
                raise ValueError("BM25 状态无法读取，请检查状态文件或恢复备份") from exc
            return
        if raw.get("version") != 1:
            if strict:
                raise ValueError("BM25 状态版本不受支持")
            return
        self._vocab = {str(k): int(v) for k, v in raw.get("vocab", {}).items()}
        self._doc_freq = Counter({str(k): int(v) for k, v in raw.get("doc_freq", {}).items()})
        self._total_docs = int(raw.get("total_docs", 0))
        self._sum_token_len = int(raw.get("sum_token_len", 0))
        if self._vocab:
            self._vocab_counter = max(self._vocab.values()) + 1
        else:
            self._vocab_counter = 0
        self._recompute_avg_len()

    def reload_state(self) -> None:
        """Refresh after acquiring process ownership; pre-lock memory may be stale."""
        with self._lock:
            self._vocab = {}
            self._vocab_counter = 0
            self._doc_freq = Counter()
            self._total_docs = 0
            self._sum_token_len = 0
            self._avg_doc_len = 1.0
            self._load_state(strict=True)

    def _persist_unlocked(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "total_docs": self._total_docs,
            "sum_token_len": self._sum_token_len,
            "vocab": self._vocab,
            "doc_freq": dict(self._doc_freq),
        }
        tmp = self._state_path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(self._state_path)

    def state_lock(self):
        """提交/补偿期间禁止查询端分配新的稀疏维度。"""
        return self._lock

    def snapshot_state(self) -> dict:
        with self._lock:
            return deepcopy({
                "version": 1, "total_docs": self._total_docs,
                "sum_token_len": self._sum_token_len, "vocab": self._vocab,
                "doc_freq": dict(self._doc_freq),
            })

    def restore_state(self, state: dict) -> None:
        """从持久事务快照恢复统计及维度映射，供提交失败和重启恢复使用。"""
        with self._lock:
            self._vocab = {str(k): int(v) for k, v in state["vocab"].items()}
            self._vocab_counter = max(self._vocab.values(), default=-1) + 1
            self._doc_freq = Counter(state["doc_freq"])
            self._total_docs = int(state["total_docs"])
            self._sum_token_len = int(state["sum_token_len"])
            self._recompute_avg_len()
            self._persist_unlocked()

    def _persist(self) -> None:
        with self._lock:
            self._persist_unlocked()

    def increment_add_documents(self, texts: list[str]) -> None:
        """
        将每个 text 视为 BM25 中的一篇文档（与当前 chunk 写入粒度一致），增量更新 N / df / 长度和。
        """
        if not texts:
            return
        with self._lock:
            for text in texts:
                tokens = self.tokenize(text)
                doc_len = len(tokens)
                self._sum_token_len += doc_len
                self._total_docs += 1
                for token in set(tokens):
                    if token not in self._vocab:
                        self._vocab[token] = self._vocab_counter
                        self._vocab_counter += 1
                    self._doc_freq[token] += 1
            self._recompute_avg_len()
            self._persist_unlocked()

    def increment_remove_documents(self, texts: list[str]) -> None:
        """
        从语料统计中移除与 increment_add_documents 对称的文档集合（如删除某文件的全部 chunk 文本）。
        词表索引不回收，避免与 Milvus 中仍可能存在的旧稀疏向量维度冲突。
        """
        if not texts:
            return
        with self._lock:
            for text in texts:
                tokens = self.tokenize(text)
                doc_len = len(tokens)
                self._sum_token_len = max(0, self._sum_token_len - doc_len)
                self._total_docs = max(0, self._total_docs - 1)
                for token in set(tokens):
                    if token not in self._doc_freq:
                        continue
                    self._doc_freq[token] -= 1
                    if self._doc_freq[token] <= 0:
                        del self._doc_freq[token]
            self._recompute_avg_len()
            self._persist_unlocked()

    def _ensure_embedder(self):
        """延迟加载 embedder"""
        if not self._embedder_ready:
            with span("initialization", component="embedding", model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"), provider="local", cold_start=True):
                self._embedder = _create_dense_embedder()
            self._embedder_ready = True

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        with span("embedding_dense", component="embedding", provider="local",
                  model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"), cold_start=not self._embedder_ready):
            self._ensure_embedder()
            try:
                return self._embedder.embed_documents(texts)
            except Exception as e:
                raise Exception(f"本地嵌入模型调用失败: {str(e)}") from e

    def tokenize(self, text: str) -> list[str]:
        text = text.lower()
        tokens = []
        chinese_pattern = re.compile(r"[\u4e00-\u9fff]")
        english_pattern = re.compile(r"[a-zA-Z]+")
        i = 0
        while i < len(text):
            char = text[i]
            if chinese_pattern.match(char):
                tokens.append(char)
                i += 1
            elif english_pattern.match(char):
                match = english_pattern.match(text[i:])
                if match:
                    tokens.append(match.group())
                    i += len(match.group())
            else:
                i += 1
        return tokens

    def _sparse_vector_for_text_unlocked(self, text: str) -> tuple[dict, bool]:
        tokens = self.tokenize(text)
        doc_len = len(tokens)
        tf = Counter(tokens)
        sparse_vector: dict[int, float] = {}
        vocab_changed = False
        n = max(self._total_docs, 0)
        avg = max(self._avg_doc_len, 1.0)

        for token, freq in tf.items():
            if token not in self._vocab:
                self._vocab[token] = self._vocab_counter
                self._vocab_counter += 1
                vocab_changed = True

            idx = self._vocab[token]
            df = self._doc_freq.get(token, 0)
            if df == 0:
                idf = math.log((n + 1) / 1)
            else:
                idf = math.log((n - df + 0.5) / (df + 0.5) + 1)

            numerator = freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / avg)
            score = idf * numerator / denominator
            if score > 0:
                sparse_vector[idx] = float(score)

        return sparse_vector, vocab_changed

    @timed("embedding_sparse", component="bm25")
    def get_sparse_embedding(self, text: str) -> dict:
        from .bm25 import BM25MigrationRequired
        raise BM25MigrationRequired("旧 BM25 查询编码已停用，请使用 Milvus 原生 BM25 文本查询")

    @timed("embedding_sparse", component="bm25")
    def get_sparse_embeddings(self, texts: list[str]) -> list[dict]:
        if not texts:
            return []
        with self._lock:
            out: list[dict] = []
            any_new_vocab = False
            for text in texts:
                sparse_vector, vocab_changed = self._sparse_vector_for_text_unlocked(text)
                out.append(sparse_vector)
                any_new_vocab = any_new_vocab or vocab_changed
            if any_new_vocab:
                self._persist_unlocked()
        return out

    def get_all_embeddings(self, texts: list[str]) -> tuple[list[list[float]], list[dict]]:
        dense_embeddings = self.get_embeddings(texts)
        sparse_embeddings = self.get_sparse_embeddings(texts)
        return dense_embeddings, sparse_embeddings


# 全进程唯一稠密嵌入实例；原生 BM25 统计由 Milvus 维护。
embedding_service = EmbeddingService()
