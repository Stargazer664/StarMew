import os

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

embedding = HuggingFaceEmbeddings(
    model_name=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
    model_kwargs={"device": os.getenv("EMBEDDING_DEVICE", "cpu")},
    encode_kwargs={"normalize_embeddings": True},
)

# 测试
vec = embedding.embed_query("测试医疗RAG")
print("向量维度:", len(vec))  # 输出 1024 就是成功！