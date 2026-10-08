"""
Pico 文档 RAG 评测脚本

使用流程：
1. 检查/上传 Pico PDF 到知识库
2. 运行 Ragas 批量评测（检索 + 生成）
3. 输出评测报告

用法: uv run python scripts/run_pico_eval.py
"""

import sys
import os
import asyncio
import json
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

# 切换到项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

from backend.evaluation import (
    evaluator, EvalSample, EvalDatasetManager, _RAGAS_AVAILABLE, _RAGAS_VERSION,
)
from backend.rag_pipeline import run_rag_graph
from backend.llm_config import create_chat_model
from rich.console import Console
from rich.table import Table

console = Console()


# ─── Step 1: 检查/上传 Pico PDF ──────────────────────────────────────

def ensure_document_indexed(filename: str) -> bool:
    """确保文档已索引到知识库"""
    from backend.milvus_client import MilvusManager

    mm = MilvusManager()
    try:
        mm.init_collection()
        results = mm.query(output_fields=["filename"], limit=10000)
        indexed_files = {r.get("filename", "") for r in results}
    except Exception as e:
        console.print(f"[yellow]⚠ 检查 Milvus 失败，尝试验证文档存在: {e}[/yellow]")
        return False

    if filename in indexed_files:
        chunk_count = sum(1 for r in results if r.get("filename") == filename)
        console.print(f"[green]✅ {filename} 已在知识库中（{chunk_count} 个分块）[/green]")
        return True

    # 文档不在知识库中，尝试上传
    pico_path = PROJECT_ROOT / "data" / "documents" / filename
    if not pico_path.exists():
        console.print(f"[red]❌ 文档不存在: {pico_path}[/red]")
        return False

    console.print(f"[yellow]📤 {filename} 未索引，正在上传...[/yellow]")
    try:
        from backend.document_loader import DocumentLoader
        from backend.parent_chunk_store import ParentChunkStore
        from backend.embedding import embedding_service
        from backend.milvus_writer import MilvusWriter

        loader = DocumentLoader()
        pcs = ParentChunkStore()
        mw = MilvusWriter(embedding_service=embedding_service, milvus_manager=mm)

        # 清理旧数据
        delete_expr = f'filename == "{filename}"'
        try:
            rows = mm.query_all(filter_expr=delete_expr, output_fields=["text"])
            texts = [r.get("text") or "" for r in rows]
            embedding_service.increment_remove_documents(texts)
        except Exception:
            pass
        try:
            mm.delete(delete_expr)
        except Exception:
            pass
        try:
            pcs.delete_by_filename(filename)
        except Exception:
            pass

        # 加载文档
        docs = loader.load_document(str(pico_path), filename)
        parent_docs = [d for d in docs if int(d.get("chunk_level", 0) or 0) in (1, 2)]
        leaf_docs = [d for d in docs if int(d.get("chunk_level", 0) or 0) == 3]

        if not leaf_docs:
            console.print("[red]❌ 文档解析失败，未生成叶子分块[/red]")
            return False

        # 写入
        pcs.upsert_documents(parent_docs)
        mw.write_documents(leaf_docs)

        console.print(f"[green]✅ 上传成功: 父级分块 {len(parent_docs)} 个, 叶子分块 {len(leaf_docs)} 个[/green]")
        return True

    except Exception as e:
        console.print(f"[red]❌ 上传失败: {e}[/red]")
        return False


# ─── Step 2: 创建评测 Pipeline（检索 + LLM 生成）─────────────────────

def create_eval_pipeline():
    """
    创建评测用 Pipeline 函数。
    返回的函数接收 question，返回 {answer, docs, rag_trace}。
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    gen_llm = create_chat_model(
        role="generation",
        temperature=0,
    )

    def eval_pipeline(question: str) -> dict:
        """检索 + 生成一体的 Pipeline"""
        # Step A: 检索
        rag_result = run_rag_graph(question)
        docs = rag_result.get("docs", [])
        rag_trace = rag_result.get("rag_trace")

        if not docs:
            return {
                "answer": "未找到相关文档，无法生成回答。",
                "docs": [],
                "rag_trace": rag_trace,
            }

        # Step B: 用检索结果生成答案
        contexts_text = "\n\n---\n\n".join([
            f"[来源: {d.get('filename', '?')}, 页码: {d.get('page_number', '?')}]\n{d.get('text', '')}"
            for d in docs[:5]
        ])

        prompt = f"""请根据以下检索到的文档内容回答问题。只基于提供的文档内容作答，不要编造信息。

【问题】
{question}

【检索到的文档内容】
{contexts_text[:8000]}

请用中文回答，尽量简洁准确。如果文档中没有相关信息，请诚实地说不知道。"""

        try:
            response = gen_llm.invoke([
                SystemMessage(content="你是一个专业的文档分析助手，基于用户提供的文档内容回答问题。"),
                HumanMessage(content=prompt),
            ])
            answer = response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            answer = f"生成答案时出错: {e}"

        return {"answer": answer, "docs": docs, "rag_trace": rag_trace}

    return eval_pipeline


# ─── Step 3: 运行评测 ──────────────────────────────────────────────

async def run_evaluation():
    console.print("\n[bold cyan]═══════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]  Hello-Agents 文档 RAG 评测 (Ragas SDK)[/bold cyan]")
    console.print(f"[bold cyan]  Ragas 版本: {_RAGAS_VERSION} | 原生模式: {_RAGAS_AVAILABLE}[/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════[/bold cyan]\n")

    # 1. 确保文档已索引
    filename = "Hello-Agents-V1.0.2-20260210.pdf"
    console.print("[1/4] 检查知识库索引...")
    if not ensure_document_indexed(filename):
        console.print("[red]无法继续：请先手动上传文档[/red]")
        return

    # 2. 加载评测数据集
    console.print("\n[2/4] 加载评测数据集...")
    dataset_path = PROJECT_ROOT / "data" / "eval_samples_hello_agents.json"
    samples = EvalDatasetManager.load_from_json(str(dataset_path))
    console.print(f"[green]  已加载 {len(samples)} 条评测问题[/green]")

    # 3. 创建 Pipeline
    console.print("\n[3/4] 创建评测 Pipeline（检索+生成）...")
    pipeline = create_eval_pipeline()

    # 4. 运行批量评测
    console.print(f"\n[4/4] 运行批量评测...")
    console.print(f"[dim]  每条样本需要: 检索 -> 生成 -> 5个Ragas指标评估[/dim]")
    console.print(f"[dim]  预计每条耗时 60-90s, 总共约 {len(samples) * 80 // 60} 分钟[/dim]\n")

    evaluator.clear_history()

    # 先测试一条，确保链路通
    test_sample = samples[0]
    console.print("[bold]🧪 预热测试...[/bold]")
    test_start = time.time()
    test_result = evaluator.evaluate_rag_pipeline(test_sample, pipeline)
    test_latency = time.time() - test_start

    console.print(f"  问题: {test_sample.question[:60]}...")
    console.print(f"  答案: {test_result.answer[:100]}...")
    overall = test_result.overall_score
    overall_text = f"{overall:.4f}" if overall is not None else "N/A"
    console.print(
        f"  综合分: {overall_text}  "
        f"耗时: {test_latency:.0f}s"
    )
    if test_result.error:
        console.print(f"  [red]错误: {test_result.error}[/red]")

    if overall is None:
        console.print("[yellow]⚠ 预热未取得有效评分，请检查评测错误。[yellow]")
    elif overall <= 0:
        console.print("[yellow]⚠ 预热测试得分异常，请检查。[yellow]")

    # 批量评测（并发数 1 避免限流）
    console.print(f"\n[cyan]▶ 开始评测全部 {len(samples)} 条样本...[/cyan]\n")
    evaluator.clear_history()  # 清除测试数据
    batch_start = time.time()

    results = await evaluator.evaluate_batch(
        samples=samples,
        rag_pipeline_fn=pipeline,
        max_concurrency=1,  # 单并发，避免 API 限流
    )

    batch_latency = time.time() - batch_start

    # ─── 输出报告 ──────────────────────────────────────────────────

    console.print("\n\n")
    evaluator.print_summary()

    # 详细结果表格
    table = Table(title="Hello-Agents RAG 评测详细结果")
    table.add_column("#", style="dim")
    table.add_column("问题", style="cyan", max_width=40)
    table.add_column("faithfulness", justify="right")
    table.add_column("answer_rel", justify="right")
    table.add_column("ctx_prec", justify="right")
    table.add_column("ctx_recall", justify="right")
    table.add_column("总评", justify="right", style="green")

    import math
    for i, r in enumerate(results):
        m = r.metrics
        f = f"{m.get('faithfulness', 0):.2f}" if m.get('faithfulness') is not None else "N/A"
        a = f"{m.get('answer_relevancy', 0):.2f}" if m.get('answer_relevancy') is not None else "N/A"
        cp = f"{m.get('context_precision', 0):.2f}" if m.get('context_precision') is not None else "N/A"
        cr = f"{m.get('context_recall', 0):.2f}" if m.get('context_recall') is not None else "N/A"
        ov = r.overall_score
        o_str = f"{ov:.2f}" if ov is not None and not (isinstance(ov, float) and math.isnan(ov)) else "N/A"
        table.add_row(str(i + 1), r.sample.question[:38], f, a, cp, cr, o_str)

    console.print(table)
    console.print(f"\n[bold]总耗时:[/bold] {batch_latency:.0f}s ({batch_latency/60:.1f}min)")
    console.print(f"[bold]平均每条:[/bold] {batch_latency/len(samples):.0f}s")

    # 导出结果
    export_path = evaluator.export_results(
        str(PROJECT_ROOT / "data" / "hello_agents_eval_results.json")
    )
    evaluator.export_csv(
        str(PROJECT_ROOT / "data" / "hello_agents_eval_results.csv")
    )
    console.print(f"\n[green]📄 结果已导出:[/green]")
    console.print(f"   JSON: {export_path}")
    console.print(f"   CSV:  data/pico_eval_results.csv")


if __name__ == "__main__":
    asyncio.run(run_evaluation())
