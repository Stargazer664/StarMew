# 跨线程 RAG 推送机制详解

## 问题背景

```
┌─────────────────────────────────────────────────────────────────┐
│                      主线程 (asyncio)                          │
│                                                                 │
│  @router.post("/chat/stream")                                  │
│  async def chat_stream_endpoint():                             │
│      async for chunk in chat_with_agent_stream():  ◀── yield  │
│          yield f"data: {...}\n\n"                              │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                              │ SSE 数据
┌─────────────────────────────┼─────────────────────────────────┐
│                    output_queue                                 │
│                    (asyncio.Queue)                             │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                              │ put_nowait()
┌─────────────────────────────┼─────────────────────────────────┐
│               工作线程 (ThreadPoolExecutor)                    │
│                                                                 │
│  react_graph.ainvoke() 运行同步代码                            │
│      │                                                         │
│      ├──► get_current_weather() ──► 同步 HTTP 请求             │
│      ├──► web_search() ──► 同步 HTTP 请求                      │
│      └──► search_knowledge_base() ──► RAG Pipeline (同步)      │
│           │                                                     │
│           ├──► emit_rag_step("🔍", "正在检索...")  ◀── 这里！   │
│           └──► 运行在 ThreadPoolExecutor 中                    │
└─────────────────────────────────────────────────────────────────┘
```

**问题**：工具函数运行在**工作线程**，但 SSE 需要向**主线程**的 asyncio 事件循环发送数据。

---

## 解决方案

使用 `call_soon_threadsafe` 实现跨线程安全调度。

---

## 核心代码

### 1. 全局变量定义

**文件**：[tools.py:16-19](../backend/tools.py#L16-L19)

```python
# RAG 步骤队列（主线程设置，工作线程使用）
_RAG_STEP_QUEUE = None  # asyncio.Queue

# 主线程的 asyncio 事件循环（捕获保存）
_RAG_STEP_LOOP = None   # asyncio.loop
```

### 2. 捕获事件循环

**文件**：[tools.py:42-53](../backend/tools.py#L42-L53)

```python
def set_rag_step_queue(queue):
    """设置 RAG 步骤队列，并捕获当前事件循环以便跨线程调度。"""
    global _RAG_STEP_QUEUE, _RAG_STEP_LOOP
    _RAG_STEP_QUEUE = queue

    if queue:
        import asyncio
        try:
            # 获取当前运行中的 asyncio 事件循环（主线程）
            _RAG_STEP_LOOP = asyncio.get_running_loop()
        except RuntimeError:
            # fallback：获取当前事件循环
            _RAG_STEP_LOOP = asyncio.get_event_loop()
    else:
        _RAG_STEP_LOOP = None
```

### 3. 跨线程推送

**文件**：[tools.py:56-65](../backend/tools.py#L56-L65)

```python
def emit_rag_step(icon: str, label: str, detail: str = ""):
    """向队列发送一个 RAG 检索步骤。支持跨线程安全调用。"""
    global _RAG_STEP_QUEUE, _RAG_STEP_LOOP

    if _RAG_STEP_QUEUE is not None and _RAG_STEP_LOOP is not None:
        step = {"icon": icon, "label": label, "detail": detail}
        try:
            if not _RAG_STEP_LOOP.is_closed():
                # 关键：call_soon_threadsafe 允许从其他线程安全地
                # 将回调调度到 asyncio 事件循环中
                _RAG_STEP_LOOP.call_soon_threadsafe(
                    _RAG_STEP_QUEUE.put_nowait,
                    step
                )
        except Exception:
            pass
```

---

## 调用链路

### Agent 设置阶段

```
agent.py - chat_with_agent_stream()
    │
    ├──► output_queue = asyncio.Queue()
    │
    ├──► class _RagStepProxy:
    │        def put_nowait(self, step):
    │            output_queue.put_nowait({"type": "rag_step", "step": step})
    │
    └──► set_rag_step_queue(_RagStepProxy())
            │
            ▼
        tools.py - set_rag_step_queue()
            │
            ├──► _RAG_STEP_QUEUE = proxy
            └──► _RAG_STEP_LOOP = asyncio.get_running_loop()
                (此时处于主线程，loop 是主线程的)
```

### 工作线程推送阶段

```
Agent 执行工具
    │
    └──► search_knowledge_base()
            │
            ▼
        rag_pipeline.py - run_rag_graph()
            │
            ├──► retrieve_documents()
            │        │
            │        ├──► emit_rag_step("🔍", "正在检索知识库...")
            │        │        │
            │        │        ▼
            │        │    tools.py - emit_rag_step()
            │        │        │
            │        │        └──► _RAG_STEP_LOOP.call_soon_threadsafe(
            │        │                 output_queue.put_nowait,  # put 方法引用
            │        │                 step                    # 参数
            │        │             )
            │        │
            │        └──► emit_rag_step("✅", "检索完成", f"找到 {n} 个片段")
            │
            └──► return result

    工作线程 ──call_soon_threadsafe──▶ 主线程事件循环
                    │
                    ▼
            output_queue.put_nowait(step)
                    │
                    ▼
            主线程 yield f"data: {json.dumps({'type': 'rag_step', 'step': step})}\n\n"
```

---

## 完整流程图

```
┌────────────────────────────────────────────────────────────────────────────┐
│                           主线程 (asyncio)                               │
│                                                                            │
│  1. 创建 Queue                                                           │
│     output_queue = asyncio.Queue()                                       │
│                                                                            │
│  2. 设置代理 + 捕获 Loop                                                  │
│     ┌──────────────────────────────────────────────────────────────┐    │
│     │ set_rag_step_queue(proxy)                                     │    │
│     │   ↓                                                          │    │
│     │ _RAG_STEP_QUEUE = proxy                                       │    │
│     │ _RAG_STEP_LOOP = asyncio.get_running_loop()  ◀── 主线程 Loop │    │
│     └──────────────────────────────────────────────────────────────┘    │
│                                                                            │
│  3. 启动后台 Agent 任务                                                   │
│     agent_task = asyncio.create_task(_agent_worker())                   │
│                                                                            │
│  4. 主循环：yield SSE 数据                                                 │
│     while True:                                                          │
│         event = await output_queue.get()  ◀── 阻塞等待                   │
│         yield f"data: {json.dumps(event)}\n\n"                           │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
                                    ▲
                                    │ call_soon_threadsafe
                                    │ put_nowait(step)
┌───────────────────────────────────┼───────────────────────────────────────┐
│            工作线程              │                                       │
│                                   │                                       │
│  5. 运行 Agent                    │                                       │
│     react_graph.ainvoke(...)     │                                       │
│         │                        │                                       │
│         ▼                        │                                       │
│  6. 执行工具                      │                                       │
│     search_knowledge_base()       │                                       │
│         │                        │                                       │
│         ▼                        │                                       │
│  7. RAG Pipeline                  │                                       │
│     run_rag_graph()               │                                       │
│         │                        │                                       │
│         ▼                        │                                       │
│  8. 发送步骤                      │                                       │
│     emit_rag_step("🔍", "检索中")  │                                       │
│         │                        │                                       │
│         ▼                        │                                       │
│  9. 跨线程调度                    │                                       │
│     _RAG_STEP_LOOP.               │                                       │
│       call_soon_threadsafe(       │                                       │
│         _RAG_STEP_QUEUE.put_nowait,│                                       │
│         step                      │                                       │
│       )                          │                                       │
│         │                        │                                       │
│         ▼                        │                                       │
│  10. 结果入队                      │                                       │
│      output_queue 收到 step       │                                       │
│          │                        │                                       │
│          └──────────────▶─────────┘                                       │
│                                     │                                     │
└─────────────────────────────────────┘                                     │
                                      │
                                      ▼
                              output_queue.get()
                              返回 step
```

---

## 为什么用 call_soon_threadsafe

### 普通线程 vs asyncio 线程

```python
# ❌ 错误：在工作线程直接调用 Queue（不安全）
def emit_rag_step(icon, label, detail):
    step = {...}
    output_queue.put_nowait(step)  # 可能导致数据竞争或死锁
```

### call_soon_threadsafe（安全）

```python
# ✅ 正确：通知主线程的 loop 执行
def emit_rag_step(icon, label, detail):
    step = {...}
    _RAG_STEP_LOOP.call_soon_threadsafe(
        _RAG_STEP_QUEUE.put_nowait,
        step
    )
```

**原理**：
- `call_soon_threadsafe` 是**线程安全**的
- 它将 `put_nowait` 调用**调度**到主线程的 asyncio 事件循环
- 下次主线程循环迭代时，才会真正执行 `put_nowait`
- 保证所有对 asyncio 对象的操作都在主线程

---

## 实际使用示例

### RAG Pipeline 中的调用

```python
# rag_pipeline.py

def retrieve_initial(state: RAGState) -> RAGState:
    query = state["question"]

    # 发送 RAG 步骤
    emit_rag_step("🔍", "正在检索知识库...", f"查询: {query[:50]}")
    #                         ↑ 图标   ↑ 标题       ↑ 详情

    retrieved = retrieve_documents(query, top_k=5)

    emit_rag_step(
        "🧱",
        "三级分块检索",
        f"L3 召回，候选 {len(retrieved.get('docs', []))}"
    )

    emit_rag_step(
        "✅",
        f"检索完成，找到 {len(results)} 个片段",
        f"模式: hybrid"
    )

    return {...}
```

### 工具执行中的调用

```python
# tools.py

@tool("get_current_weather")
def get_current_weather(location: str) -> str:
    emit_rag_step("🔧", "正在执行工具: get_current_weather", f"location={location}")

    # ... 执行逻辑 ...

    emit_rag_step("✅", "工具 get_current_weather 执行完成", result[:80])
    return result
```

---

## 前端显示效果

```
┌─────────────────────────────────────────────────────┐
│  🔍 正在检索知识库...                              │
│      查询: 什么是 RAG...                           │
│                                                     │
│  🧱 三级分块检索                                   │
│      L3 召回，候选 15                              │
│                                                     │
│  🧩 Auto-merging 合并                             │
│      启用: True，应用: True                        │
│                                                     │
│  ✅ 检索完成，找到 5 个片段                        │
│      模式: hybrid                                 │
│                                                     │
│  ─────────────────────────────────────────────     │
│                                                     │
│  RAG（Retrieval-Augmented Generation）是一种...    │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## 关键点总结

| 要点 | 说明 |
|------|------|
| **保存 Loop** | `asyncio.get_running_loop()` 在主线程保存 |
| **线程安全** | `call_soon_threadsafe` 是 asyncio 官方推荐的跨线程方式 |
| **延迟执行** | 回调在下一次事件循环迭代时执行 |
| **异常忽略** | `try-except` 避免推送失败影响主流程 |
| **Loop 检测** | `is_closed()` 避免向已关闭的 loop 发送 |

---

## 简历关键词

```
跨线程通信, asyncio, call_soon_threadsafe,
工作线程, 主线程, 事件循环, 线程安全,
LangGraph, ReAct Agent
```

---

*文档更新时间：2026-05-14*