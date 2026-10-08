"""Reindex Hello-Agents PDF with text normalization."""
import sys
import os
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from backend.document_loader import DocumentLoader
from backend.embedding import embedding_service
from backend.milvus_client import MilvusManager
from backend.milvus_writer import MilvusWriter
from backend.parent_chunk_store import ParentChunkStore

filename = "Hello-Agents-V1.0.2-20260210.pdf"
print(f"Step 1: Loading {filename} with normalization...", flush=True)
loader = DocumentLoader()
docs = loader.load_document(
    str(PROJECT_ROOT / "data" / "documents" / filename),
    filename,
)
parent_docs = [d for d in docs if int(d.get("chunk_level", 0) or 0) in (1, 2)]
leaf_docs = [d for d in docs if int(d.get("chunk_level", 0) or 0) == 3]
print(f"  Loaded: {len(parent_docs)} parent, {len(leaf_docs)} leaf chunks", flush=True)

print("Step 2: Init Milvus...", flush=True)
mm = MilvusManager()
mm.init_collection()
pcs = ParentChunkStore()

# Clean old
print("Step 3: Clean old data...", flush=True)
try:
    rows = mm.query_all(
        filter_expr=f'filename == "{filename}"',
        output_fields=["text"],
    )
    if rows:
        print(f"  Found {len(rows)} old chunks, deleting...", flush=True)
        mm.delete(f'filename == "{filename}"')
    else:
        print("  No old data", flush=True)
except Exception as e:
    print(f"  Cleanup error: {e}", flush=True)

print("Step 4: Write to PostgreSQL...", flush=True)
pcs.upsert_documents(parent_docs)
print(f"  Wrote {len(parent_docs)} parent chunks", flush=True)

print(f"Step 5: Write {len(leaf_docs)} leaf chunks to Milvus (this may take 5-10 min)...", flush=True)
mw = MilvusWriter(embedding_service=embedding_service, milvus_manager=mm)

def progress_cb(p, t):
    if p % 200 == 0 or p == t or p == 0:
        print(f"  {p}/{t} ({100*p//t}%)", flush=True)

mw.write_documents(leaf_docs, progress_callback=progress_cb)
print(f"\n✅ Done: {len(leaf_docs)} chunks reindexed", flush=True)
