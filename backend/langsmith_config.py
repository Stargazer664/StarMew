"""
LangSmith 追踪配置模块

启用方式：
1. 在 .env 中配置 LANGCHAIN_API_KEY 和 LANGCHAIN_PROJECT
2. 本模块会在后端启动时自动初始化
3. 登录 https://smith.langchain.com 查看追踪记录
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# 加载 .env
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)

# LangSmith 配置
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "starmew-dev")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
LANGCHAIN_DEBUG = os.getenv("LANGCHAIN_DEBUG", "false").lower() == "true"


def setup_langsmith():
    """初始化 LangSmith 追踪（不阻塞）"""
    if os.getenv("LLM_PROVIDER", "ollama").strip().lower() == "ollama":
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        os.environ["LANGSMITH_TRACING"] = "false"
        return
    if not LANGCHAIN_API_KEY:
        return

    try:
        # 只设置环境变量，不创建 Client（避免网络阻塞）
        os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
        os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT

        if LANGCHAIN_TRACING_V2:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"

        if LANGCHAIN_DEBUG:
            os.environ["LANGCHAIN_DEBUG"] = "true"

        print(f"[LangSmith] Enabled - Project: {LANGCHAIN_PROJECT}")

    except Exception as e:
        print(f"[LangSmith] Warning: {e}")


def get_langsmith_tags() -> list:
    """获取 LangSmith 追踪标签"""
    return ["starmew", "react-agent", "rag"]


# 自动初始化
setup_langsmith()
