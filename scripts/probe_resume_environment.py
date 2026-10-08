"""Read-only local environment/corpus inventory for the resume benchmark."""
from __future__ import annotations

import hashlib
import argparse
import json
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from dotenv import dotenv_values
from pymilvus import MilvusClient


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=root / 'data' / 'resume_benchmark_20261004' / 'environment.json')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists; choose a new snapshot path')
    cfg = dotenv_values(root / '.env')
    name = cfg.get('MILVUS_COLLECTION', 'embeddings')
    client = MilvusClient(uri=f"http://{cfg.get('MILVUS_HOST', '127.0.0.1')}:{cfg.get('MILVUS_PORT', '19530')}")
    rows = []
    iterator = client.query_iterator(collection_name=name, filter='', output_fields=['filename', 'chunk_id', 'chunk_level'], batch_size=1000)
    try:
        while batch := iterator.next():
            rows.extend(batch)
    finally:
        iterator.close()
        client.close()
    names = Counter(row.get('filename', '') for row in rows)
    payload = {
        'observed_at': datetime.now(timezone.utc).isoformat(),
        'python': platform.python_version(), 'platform': platform.platform(),
        'collection': name, 'indexed_chunks': len(rows),
        'indexed_documents': len(names), 'filenames': dict(sorted(names.items())),
        'levels': dict(Counter(str(row.get('chunk_level')) for row in rows)),
        'index_identity_sha256': hashlib.sha256(json.dumps(sorted((row.get('chunk_id', ''), row.get('filename', '')) for row in rows), ensure_ascii=False).encode()).hexdigest(),
        'model': cfg.get('OLLAMA_MODEL'), 'embedding_model': cfg.get('EMBEDDING_MODEL'),
        'embedding_device': cfg.get('EMBEDDING_DEVICE'),
        'note': '只读索引标识快照，不证明索引正文与本地文件逐字相同。',
    }
    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in payload.items() if k != 'filenames'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
