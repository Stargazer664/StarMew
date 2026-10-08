"""对 dishes 评测集执行 BM25 稀疏召回审计，避免本地 Torch 嵌入不可用时误报 Ragas 分数。"""

from __future__ import annotations

import json
import time
from pathlib import Path

from backend.bm25 import lexical_text
from backend.milvus_client import MilvusManager
from backend.evaluation import EvalDatasetManager


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "eval_samples_dishes.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "eval_dishes_sparse_audit.json"


def sparse_retrieve(manager: MilvusManager, query: str, top_k: int = 5) -> list[dict]:
    manager.assert_bm25_compatible()
    sparse = lexical_text(query)
    if not sparse:
        return []
    client = manager._get_client()
    rows = client.search(
        collection_name=manager.collection_name,
        data=[sparse],
        anns_field="sparse_embedding",
        search_params={"metric_type": "BM25", "params": {}},
        limit=top_k,
        output_fields=["filename", "text", "chunk_level", "chunk_id"],
        consistency_level="Strong",
    )
    return [
        {
            "filename": hit.get("entity", {}).get("filename", ""),
            "text": hit.get("entity", {}).get("text", ""),
            "chunk_level": hit.get("entity", {}).get("chunk_level", 0),
            "chunk_id": hit.get("entity", {}).get("chunk_id", ""),
            "score": hit.get("distance", 0.0),
        }
        for hit in (rows[0] if rows else [])
    ]


def main() -> None:
    samples = EvalDatasetManager.load_from_json(str(DATASET_PATH))
    manager = MilvusManager()
    manager.init_collection()
    records = []
    hit1 = hit3 = hit5 = 0
    answerable = unanswerable = 0
    unanswerable_nonempty = 0
    started = time.time()

    for idx, sample in enumerate(samples, 1):
        metadata = sample.metadata or {}
        is_unanswerable = bool(metadata.get("unanswerable"))
        expected = f"{metadata.get('dish', '')}.md" if not is_unanswerable else ""
        try:
            docs = sparse_retrieve(manager, sample.question, top_k=5)
            filenames = [doc.get("filename", "") for doc in docs]
            if is_unanswerable:
                unanswerable += 1
                if filenames:
                    unanswerable_nonempty += 1
            else:
                answerable += 1
                hit1 += int(bool(filenames and filenames[0] == expected))
                hit3 += int(expected in filenames[:3])
                hit5 += int(expected in filenames[:5])
            records.append({
                "index": idx,
                "question": sample.question,
                "expected_filename": expected or None,
                "unanswerable": is_unanswerable,
                "retrieved_filenames": filenames,
                "retrieved_docs_count": len(docs),
                "top_score": docs[0].get("score") if docs else None,
                "error": None,
            })
        except Exception as exc:
            records.append({
                "index": idx,
                "question": sample.question,
                "expected_filename": expected or None,
                "unanswerable": is_unanswerable,
                "retrieved_filenames": [],
                "retrieved_docs_count": 0,
                "top_score": None,
                "error": str(exc),
            })
        if idx == 1 or idx % 10 == 0 or idx == len(samples):
            print(f"[{idx}/{len(samples)}] 已完成，命中@1={hit1}、命中@5={hit5}", flush=True)

    summary = {
        "total": len(samples),
        "answerable": answerable,
        "unanswerable": unanswerable,
        "hit_at_1": round(hit1 / answerable, 4) if answerable else 0.0,
        "hit_at_3": round(hit3 / answerable, 4) if answerable else 0.0,
        "hit_at_5": round(hit5 / answerable, 4) if answerable else 0.0,
        "unanswerable_nonempty_rate": round(unanswerable_nonempty / unanswerable, 4) if unanswerable else 0.0,
        "errors": sum(1 for item in records if item.get("error")),
        "elapsed_seconds": round(time.time() - started, 2),
        "retrieval_mode": "sparse_bm25",
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    print(f"结果已保存: {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
