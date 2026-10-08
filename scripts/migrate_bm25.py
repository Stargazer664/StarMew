"""Run from SuperMew: python scripts/migrate_bm25.py --target NEW [--activate]."""
import argparse
import json
import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="重建原生 BM25；保留旧集合，核对通过后可切换配置")
    parser.add_argument("--source", help="默认读取 MILVUS_COLLECTION")
    parser.add_argument("--target", required=True, help="必须是尚不存在的新集合名")
    parser.add_argument("--activate", action="store_true", help="核对成功后更新 .env 中的集合名")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()
    from dotenv import load_dotenv
    load_dotenv(args.env_file)
    from backend.bm25_migration import migrate, activate
    from backend.document_transactions import assert_index_available
    from backend.milvus_client import MilvusManager
    from backend.process_guard import backend_process_lock
    manager = MilvusManager()
    source = args.source or manager.collection_name
    try:
        with backend_process_lock():
            assert_index_available()
            print("[1/3] 检查 Milvus 连接及迁移条件", flush=True)
            with socket.create_connection((manager.host, int(manager.port)), timeout=3):
                pass
            if args.activate and os.getenv("MILVUS_COLLECTION", source) != source:
                raise ValueError("进程环境 MILVUS_COLLECTION 与迁移源不一致，请先统一部署配置")
            print(f"[2/3] 复制 {source} → {args.target}；旧集合保留", flush=True)
            report = migrate(manager._get_client(), source, args.target, batch_size=args.batch_size,
                             progress=lambda message: print(message, flush=True))
            print("[3/3] 全量核对通过", flush=True)
            if args.activate:
                backup = activate(report, args.env_file)
                print(f"已更新配置，原配置备份：{backup}。请重启后端并确保外部环境变量指向新集合。", flush=True)
            else:
                print("未切换配置；确认后设置 MILVUS_COLLECTION 为新集合并重启后端。", flush=True)
            print(json.dumps(report, ensure_ascii=False), flush=True)
    except Exception as exc:
        print(f"迁移停止：{exc}。旧集合保留；若已创建目标集合，请检查后使用另一新集合名重试。", file=sys.stderr)
        return 1
    finally:
        manager._close_client(manager.client)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
