#!/usr/bin/env python
"""StarMew 启动脚本 - 分步导入"""
import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["WANDB_DISABLED"] = "true"

# 分步导入，避免 Windows DLL 冲突
print("Step 1/5 - api module")
from backend import api as api_module

print("Step 2/5 - database")
from backend.database import init_db

print("Step 3/5 - app")
from backend.app import app

print("Step 4/5 - uvicorn")
import uvicorn

port = int(os.getenv("PORT", 8000))
print(f"Step 5/5 - Starting on http://127.0.0.1:{port}")

uvicorn.run(app, host="0.0.0.0", port=port)
