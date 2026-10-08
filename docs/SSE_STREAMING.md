# StarMew 流式 SSE 输出设计

## 2026-10-06：中断与补充信息

当前实现由 `conversation_turns.py` 持有运行生命周期，`run_control.py` 将取消传播到异步任务和同步工具的检查点。下方旧代码示例用于说明 SSE 基础结构；实际保存与取消行为以本节和当前源码为准。

### 消息与运行状态

- `request_id` 是公开 `run_id`。内部身份同时包含会话 ID、会话实例 UUID 和 turn ID，防止删除重建后旧任务写入新会话。
- 接收请求时保存用户消息；完整回答生成后单独提交助手消息、摘要检查点和记忆任务。当前用户消息在提示词中只出现一次。
- 中断时保存已输出的部分回答，标记 `interrupted`；失败部分标记 `failed`。这些助手消息在历史界面可见，但不进入下一轮推理、摘要或长期记忆提取。
- 普通运行状态为 `running` → `completed` / `failed`；取消为 `running` → `cancel_requested` → `cancelled`。补充先保存为 `queued`，旧运行真正清理完成后才可以转为 `running`。
- 取消和完成提交锁定同一会话实体，先提交的状态生效。回答已提交时，取消返回 `completed` 和完整回答，前端保留该回答。
- 运行结束原因区分主动停止（`REQUEST_CANCELLED`）与连接断开（`CONNECTION_INTERRUPTED`）；二者都保留已经接受的用户消息。

### API

所有接口沿用 JWT 认证，并验证运行属于当前用户和指定会话。

```http
POST /api/chat/runs/{run_id}/cancel
Content-Type: application/json

{"session_id":"session_1"}
```

补充信息使用同一接口，在同一事务保存补充并请求取消旧运行：

```json
{
  "session_id": "session_1",
  "supplement_message": "不能吃鸡蛋，请推荐无蛋配方",
  "supplement_request_id": "2a6a3026-d097-47f9-b0bb-8c667ce95f88"
}
```

响应包含 `run_id`、`status`、`next_run_id`。请求可重复提交，同 UUID、同输入不会重复保存补充；变更输入会返回冲突。`GET /api/chat/runs/{run_id}?session_id=session_1` 用于确认实际终态。

前端在生成期间可以输入补充并点击发送。它在收到保存确认之后关闭旧 SSE，等待旧运行终态，然后以补充 UUID 调用原 `/api/chat/stream`。如果补充已提交但确认响应丢失，前端保留 UUID，再次确认时重用同一事务；不会根据等待时间推断工具已经退出。

`GET /api/sessions/{session_id}` 的消息带 `status`，并提供可选 `pending_run`：`run_id`、`source_run_id`、`message`。重新打开会话后可继续已保存的排队补充。服务进程异常重启时按既有独占进程恢复流程将未结束运行标记失败，用户消息仍保留。

所有 JSON SSE 事件带 `run_id`，结束哨兵仍为 `data: [DONE]`。前端同时校验 run ID、会话和流代次，迟到的事件或历史请求不能更新另一轮运行。

### 工具限制与审计

同步工具线程不能被 `Task.cancel()` 强制终止。每次工具调用前后、检索与重排等步骤之间检查共享取消标志；已启动的网络调用需等待返回或超时。旧线程退出前始终保留会话占用，后续工具不会继续执行。

工具执行另存 `tool_executions`，包含运行、调用 ID、参数、实际结果和取消标记。取消后仍成功返回的工具记为 `completed`，不能伪装为已取消；进程中断时尚未确认的工具记为 `unknown`。这些结果不写入普通聊天历史，也不自动作为新运行证据。补充后重新规划和查询，当前未实现 checkpoint 恢复或跨运行结果缓存。

启动时 `init_db()` 自动增量添加消息状态、补充父请求及会话实例字段，并创建工具记录表；保留原消息，不执行清空或替换。现有后端进程需要重启以完成 schema 初始化，浏览器需刷新加载新脚本。

### 验证记录

2026-10-06：相关回归测试 383 项、子测试 12 项通过；7 项需显式启用隔离 PostgreSQL 或 Redis 的测试跳过。旧表增量迁移已在隔离 SQLite 上验证，真实 PostgreSQL 的迁移与并发仍需部署环境验证。前端 Node 语法检查、后端 Python 编译检查及 `git diff --check` 通过。

## 概述

项目使用 **Server-Sent Events (SSE)** 实现流式输出，支持：
- Agent 思考步骤实时推送
- 模型生成 token 实时推送
- RAG 检索过程可视化

---

## 架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                         前端 (Vue 3)                            │
│                                                                 │
│  EventSource ───────────────────────────────────────────────▶   │
│  (SSE 连接)                         ◀──────────────────────── │
│                                      SSE 数据流                 │
│                                                                 │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────────────┐  │
│  │ RAG 步骤卡  │   │ Token 流输出 │   │   检索结果展示      │  │
│  │ 🔍 检索中   │   │ 实时生成    │   │   可点击查看原文    │  │
│  │ ✅ 完成    │   │ 显示...     │   │                     │  │
│  └─────────────┘   └─────────────┘   └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │ SSE
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI 后端                               │
│                                                                 │
│  @router.post("/chat/stream")                                  │
│  return StreamingResponse(event_generator(),                    │
│      media_type="text/event-stream")                            │
│                                                                 │
│  async def event_generator():                                   │
│      async for chunk in chat_with_agent_stream():              │
│          yield f"data: {json.dumps(chunk)}\n\n"                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Agent 层                                  │
│                                                                 │
│  async def chat_with_agent_stream():                           │
│      │                                                         │
│      ├──► _RagStepProxy.put_nowait(step)                       │
│      │    跨线程推送到 output_queue                             │
│      │                                                         │
│      └──► _agent_worker() 后台任务                             │
│           ├──► await react_graph.ainvoke()                      │
│           ├──► model.stream() 逐 token 推送                     │
│           └──► output_queue.put(chunk)                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 核心代码解析

### 1. API 层 - SSE 端点

**文件**：[api.py:217-238](../backend/api.py#L217-L238)

```python
@router.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest, current_user: User = Depends(get_current_user)):
    """跟 Agent 对话 (流式)"""

    async def event_generator():
        try:
            session_id = request.session_id or "default_session"
            async for chunk in chat_with_agent_stream(request.message, current_user.username, session_id):
                yield chunk  # yield SSE 数据
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用 Nginx 缓冲
        },
    )
```

### 2. Agent 层 - 流式生成器

**文件**：[agent.py:470-592](../backend/agent.py#L470-L592)

```python
async def chat_with_agent_stream(user_text, user_id, session_id):
    """使用 ReAct Agent 处理用户消息并流式返回响应"""

    # 1. 加载历史消息和记忆
    messages = storage.load(user_id, session_id)
    memory_context = memory_service.get_memory_for_context(user_id)

    # 2. 创建统一输出队列
    output_queue = asyncio.Queue()

    # 3. RAG 步骤代理（跨线程安全推送）
    class _RagStepProxy:
        def put_nowait(self, step):
            output_queue.put_nowait({"type": "rag_step", "step": step})

    set_rag_step_queue(_RagStepProxy())

    # 4. 后台任务运行 Agent
    async def _agent_worker():
        result = await react_graph.ainvoke({
            "messages": messages,
            "step_count": 0,
            "tool_output": None,
            "memory_context": memory_context,
            "stream_queue": output_queue,
            "stream_loop": asyncio.get_running_loop(),
        })

        # 提取最终响应
        content = result["messages"][-1].content
        agent_result_holder["content"] = content

        # 模型生成期间，推理节点已通过 model.stream() 将每个 token
        # 投递到 output_queue；此处只提取完整答案用于存储。

        await output_queue.put(None)  # 结束信号

    # 5. 启动后台任务
    agent_task = asyncio.create_task(_agent_worker())

    # 6. 主循环：yield SSE 数据
    try:
        while True:
            event = await output_queue.get()
            if event is None:
                break
            yield f"data: {json.dumps(event)}\n\n"
    except GeneratorExit:
        agent_task.cancel()
        raise

    # 7. 发送 RAG trace
    rag_trace = get_last_rag_context(clear=True).get("rag_trace")
    if rag_trace:
        yield f"data: {json.dumps({'type': 'trace', 'rag_trace': rag_trace})}\n\n"

    # 8. 发送结束信号
    yield "data: [DONE]\n\n"
```

---

## SSE 事件类型

| 类型 | 触发时机 | 数据结构 |
|------|----------|----------|
| `rag_step` | 工具执行中 | `{"icon": "🔍", "label": "正在检索", "detail": "..."}` |
| `content` | 最终回复 | `{"type": "content", "content": "片段"}` |
| `trace` | 检索完成 | `{"type": "trace", "rag_trace": {...}}` |
| `error` | 发生错误 | `{"type": "error", "content": "错误信息"}` |
| `[DONE]` | 结束 | 无数据 |

---

## 跨线程推送机制

### 问题

Agent 在线程池中执行工具（同步代码），但 SSE 需要在主线程的 asyncio 循环中 yield。

### 解决方案

```python
# tools.py - 全局变量
_RAG_STEP_QUEUE = None
_RAG_STEP_LOOP = None  # 保存主线程的 asyncio loop

def set_rag_step_queue(queue):
    """主线程调用：捕获当前 event loop"""
    global _RAG_STEP_QUEUE, _RAG_STEP_LOOP
    _RAG_STEP_QUEUE = queue
    if queue:
        _RAG_STEP_LOOP = asyncio.get_running_loop()

def emit_rag_step(icon, label, detail):
    """工具中调用：跨线程安全推送"""
    if _RAG_STEP_QUEUE and _RAG_STEP_LOOP:
        step = {"icon": icon, "label": label, "detail": detail}
        # call_soon_threadsafe 允许从其他线程安全地调用 asyncio
        _RAG_STEP_LOOP.call_soon_threadsafe(
            _RAG_STEP_QUEUE.put_nowait,
            step
        )
```

### 流程图

```
主线程 (asyncio)                工作线程 (ThreadPoolExecutor)
┌─────────────────┐             ┌─────────────────┐
│ set_rag_step_   │────────────▶│ _RAG_STEP_LOOP  │
│ queue(proxy)    │             │ 已保存 loop     │
└─────────────────┘             └─────────────────┘
                                      │
                                      │ emit_rag_step("🔍", "检索中")
                                      ▼
                              ┌─────────────────┐
                              │ call_soon_      │
                              │ threadsafe(     │
                              │   put_nowait,   │
                              │   step          │
                              │ )               │
                              └────────┬────────┘
                                       │ 安全推送
                                       ▼
                              ┌─────────────────┐
                              │ output_queue    │
                              │.put_nowait()    │
                              └────────┬────────┘
                                       │
                                       ▼
                              ┌─────────────────┐
                              │ 主线程 yield    │
                              │ SSE 数据        │
                              └─────────────────┘
```

---

## 前端处理示例

```javascript
// Vue 3 前端
const handleStream = async (message) => {
  const response = await fetch('/chat/stream', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`
    },
    body: JSON.stringify({ message, session_id: currentSessionId })
  })

  const reader = response.body.getReader()
  const decoder = new TextDecoder()

  // 状态管理
  let fullContent = ''
  let ragSteps = []

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    const text = decoder.decode(value)
    const lines = text.split('\n')

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue

      const data = JSON.parse(line.slice(6))

      if (data.type === 'rag_step') {
        // RAG 步骤卡片
        ragSteps.push(data.step)
        updateRagSteps(ragSteps)
      }
      else if (data.type === 'content') {
        // 追加模型刚生成的 token chunk
        fullContent += data.content
        updateMessage(fullContent)
      }
      else if (data.type === 'trace') {
        // 检索结果
        showRagTrace(data.rag_trace)
      }
      else if (data.type === 'error') {
        showError(data.content)
      }
    }
  }
}
```

---

## SSE 响应格式

```
data: {"type":"rag_step","step":{"icon":"🤔","label":"ReAct Agent 开始思考...","detail":""}}

data: {"type":"rag_step","step":{"icon":"🔧","label":"正在执行工具: search_knowledge_base","detail":"{'query': '...'}"}}

data: {"type":"rag_step","step":{"icon":"🔍","label":"正在检索知识库...","detail":"查询: 什么是 RAG..."}}

data: {"type":"rag_step","step":{"icon":"✅","label":"检索完成","detail":"找到 5 个片段"}}

data: {"type":"content","content":"R"}

data: {"type":"content","content":"AG"}

data: {"type":"content","content":"是一"}

data: {"type":"content","content":"种结"}

...

data: {"type":"trace","rag_trace":{"tool_used":true,"retrieved_chunks":[...]}}

data: [DONE]
```

---

## 性能优化

| 优化项 | 说明 |
|--------|------|
| **批量推送** | RAG 步骤实时推送，无需等待 |
| **模型流式输出** | `model.stream()` 生成一个 chunk 即推送一个 SSE 事件 |
| **Nginx 配置** | `X-Accel-Buffering: no` 禁用缓冲 |
| **线程安全** | `call_soon_threadsafe` 跨线程推送 |
| **取消处理** | `GeneratorExit` 时取消后台任务 |

---

## 简历关键词

```
SSE, Server-Sent Events, 流式输出, Streaming,
跨线程推送, asyncio.Queue, call_soon_threadsafe,
模型 token 流, 实时步骤推送, StreamingResponse
```

---

*文档更新时间：2026-05-14*
