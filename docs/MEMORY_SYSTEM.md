# StarMew 记忆系统设计

## 概述

项目采用**短期记忆 + 长期记忆**双层设计：

```
┌─────────────────────────────────────────────────────────────┐
│                      记忆系统                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           短期记忆（会话级）                          │   │
│  │  ConversationStorage                                 │   │
│  │  - 当前会话消息                                      │   │
│  │  - PostgreSQL 权威读写、事务追加                     │   │
│  │  - 支持多会话（每个 session_id 独立）                 │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                │
│                           ▼                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           长期记忆（用户级）                          │   │
│  │  UserMemoryService                                   │   │
│  │  - 跨会话累积的事实/偏好                             │   │
│  │  - 存储在 user_memories 表                           │   │
│  │  - 话题检测、会话摘要                                 │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 一、短期记忆设计

### 1.1 数据结构

```python
# 存储位置：PostgreSQL（消息和列表不读取 Redis）

# PostgreSQL 表：chat_sessions
ChatSession:
    id (PK)
    user_id (FK)
    session_id           # 会话标识，如 "default_session"
    metadata_json       # 元数据
    updated_at
    created_at

# PostgreSQL 表：chat_messages
ChatMessage:
    id (PK)
    session_ref_id (FK)  # 关联 session
    message_type         # "human" | "ai" | "system"
    content              # 消息内容
    timestamp
    rag_trace            # RAG 追踪信息
```

### 1.2 存储逻辑

**文件**：[agent.py](../backend/agent.py) 中的 `ConversationStorage` 类

```python
storage.append_turn(user_id, session_id, user_text, answer,
                    context_state=context_state, rag_trace=rag_trace,
                    request_id=lease.request_id, turn_identity=lease.identity)
# 同一事务：追加两条消息 + 更新上下文 + 完成 ChatTurn + 写 MemoryUpdateJob。
# 失败全体回滚；已有消息 ID、时间戳、正文和引用不重写。
messages = storage.load(user_id, session_id)
# 一次规范化历史查询，转为 LangChain 消息；不读/回填旧会话缓存。
```

### 1.3 读写流程

加载：按用户与会话查询 PostgreSQL → 按消息 ID 排序 → 返回规范化消息。会话列表也直接查询数据库，提交后的新读取不会被旧 Redis 值覆盖；提交之前已开始的读取可以返回先前快照。

保存：会话请求准入 → 不持数据库锁地生成回答 → 短事务追加及入队 → 提交 → 尝试删除旧消息/列表缓存键 → 返回成功。Redis 清理失败不改变已提交结果，也不会在连接恢复或进程重启后重新信任旧值。代价是增加数据库读取；未验证高并发生产吞吐。

`request_id` 在同一会话内去重：完成请求可重放，运行中的同会话请求返回 HTTP 409，ID 相同但输入不同会拒绝。旧客户端不传 ID 时仍可用，但跨请求重试无法保证去重。`replace_all_messages` 仅用于停机维护，不能用于正常聊天。

准入产生不可变 `TurnIdentity(session_ref_id, turn_id, request_id)`，前台加载、摘要状态读取及追加都沿用这份身份。删除运行中的会话后，旧生成不能写入同名重建实体：同步返回 HTTP 409 / `REQUEST_ENTITY_GONE`，已开始的 SSE 返回同码 error 后结束；此前可能已发送的答案片段不表示提交成功。旧 lease 清理也仅匹配原身份。未传 request_id 的客户端由服务端生成 UUID 后走相同流程。底层带 request_id 的追加必须同时提供准入身份，不能只凭名称和 UUID 再查一轮。

### 1.4 多会话支持

```python
# 每个用户可以创建多个会话
session_ids = storage.list_sessions(user_id)
# → ["default_session", "project_alpha", "meeting_notes", ...]

# 切换会话加载不同上下文
messages_a = storage.load(user_id, "project_alpha")
messages_b = storage.load(user_id, "meeting_notes")
```

### 1.5 增量摘要与 token 预算

`conversation_context.py` 按 token 预算选择近期完整轮次，超过历史预算时把新淘汰的旧消息并入已有摘要。摘要调用也分批限制输入；不再按“50 条触发、前 40 条摘要”处理。

`chat_sessions.metadata_json.context_state` 保存摘要、已处理消息数及历史前缀校验值。成功保存本轮回答时一并保存进度，下轮只摘要游标之后的新旧消息；服务重启后继续复用。摘要失败或回答取消时不推进进度，原始历史完整保留。历史前缀被修改时，旧摘要校验失败并重新生成。

| 配置 | 默认值 | 含义 |
|---|---:|---|
| `CHAT_CONTEXT_WINDOW_TOKENS` | 8192 | 应配置为服务端实际模型上下文窗口 |
| `CHAT_OUTPUT_RESERVE_TOKENS` | 2048 | 为输出预留，并作为聊天生成 `max_tokens` |
| `CHAT_HISTORY_MAX_TOKENS` | 3072 | 未摘要近期历史的预算 |
| `CHAT_SUMMARY_MAX_TOKENS` | 768 | 滚动摘要上限 |
| `CHAT_MEMORY_MAX_TOKENS` | 768 | 注入的长期记忆上限 |
| `CHAT_TOKENIZER_PATH` | 空 | 可选的生成模型本地 tokenizer 目录 |

配置本地 tokenizer 时仅 generation 角色按其编码计数，不联网下载；独立 router、judge、fast 始终使用保守 UTF-8 字节估算，避免共用错误 tokenizer。消息协议另计余量，估算不等同于 provider usage。输出 cap 取模型原上限与预留的较小值。窗口必须与模型实际服务配置一致；4096/2048 可能不足以容纳固定 system/schema，4096/1024 也须按实际提示和模型验证，不能默认保证可用。

每个实际模型调用（决策、最终生成、评分、重写、摘要、记忆）检查完整输入及结构化 schema。分配器保留当前问题、system、调用 ID/返回配对，限制记忆/摘要/历史并为证据保留空间；证据压缩按完整正文句与来源映射，标题字段不能独占补充配额，表格行和 URL 不从中间截断。必要输入仍超预算时明确报错，不截断当前问题，也不通过路由回退绕过预算。原始候选与最终 `answer_evidence` 并存；来源数和 `omitted_count` 不代表事实覆盖或语义正确性。

同步响应、SSE（含重放）和历史 GET 共用公开证据投影：有 `answer_evidence` 时只展示选入正文及来源、citation_id、页码和省略数量；空列表表示明确未选入，不能回退展示候选。旧 trace 缺该字段时保留旧来源展示，公开数据始终递归移除内部查询，完整原始候选仍可保存在服务端诊断。

---

## 二、长期记忆设计

### 2.1 数据结构

**文件**：[memory.py](../backend/memory.py) 中的 `UserMemoryService`

```python
# PostgreSQL 表：user_memories
UserMemory:
    id (PK)
    user_id (FK, unique)
    
    # 用户画像
    facts: dict              # {"user_name": "张三", "company": "某公司", ...}
    preferences: dict       # {"food_preference": "川菜", ...}
    
    # 会话摘要
    session_summaries: dict # {"2026-05-14": "讨论了 RAG 实现", ...}
    
    # 话题检测
    detected_topics: list    # ["RAG", "Agent", "LangChain", ...]
    
    # 统计
    total_sessions: int
    total_messages: int
    last_updated
    created_at
```

### 2.2 事实提取（两层）

```
用户消息
    │
    ├──► 正则提取 ─────────────────────────► facts
    │     (快速，轻量)
    │     例："我叫张三" → {"user_name": "张三"}
    │
    └──► LLM 提取 ───────────────────────────► facts
          (复杂，需 API 调用)
          例：从对话推断用户职位、公司等
```

### 2.3 话题检测

```python
# 关键词匹配检测
topic_keywords = {
    "RAG": ["RAG", "检索增强", "rag"],
    "Agent": ["Agent", "智能体", "agent"],
    "LangChain": ["LangChain", "langchain"],
    "向量数据库": ["向量", "Milvus", "embedding"],
    "LLM": ["LLM", "大模型", "GPT"],
    ...
}
```

---

## 三、记忆上下文注入

### 3.1 短期记忆注入

```python
# agent.py - 对话加载时
messages = storage.load(user_id, session_id)
# → [HumanMessage, AIMessage, HumanMessage, AIMessage, ...]

# 按 token 选择新增待摘要历史，合并已有摘要
state = storage.load_context_state(user_id, session_id)
plan = plan_compaction(messages, state, budget)
summary = summarize_old_messages(model, plan["pending"], plan["summary"], budget)
# 模型输入由摘要、近期完整轮次和当前问题组成；不把摘要写入原始消息表
```

### 3.2 长期记忆注入

```python
# agent.py - 获取记忆上下文
memory_context = memory_service.get_memory_for_context(user_id)
# → "【用户记忆】
#     已知信息：
#     - 姓名：张三
#     - 公司：某公司
#    关注领域：RAG, Agent
#    近期对话：
#    - 2026-05-14：讨论了 RAG 实现"

# 按预算裁剪后，仅注入一次 system prompt
system_content = REACT_SYSTEM_PROMPT.format(memory_context=memory_context)
full_messages = [SystemMessage(content=system_content)] + messages
```

### 3.3 System Prompt 模板

```python
REACT_SYSTEM_PROMPT = """你是一只可爱的猫咪助手...

【用户记忆】
{memory_context}  ← 长期记忆在这里注入

规则：
- 如果检索到的信息不足，请诚实地说不知道。
- 回答时不要暴露推理过程。
"""
```

---

## 四、记忆更新流程

```
用户发送消息
    │
    ▼
┌─────────────────┐
│ Agent 处理消息   │
└────────┬────────┘
         │
         ├─► 工具执行（天气/搜索）
         │
         ▼
┌─────────────────┐
│ 保存对话        │
│ (短期记忆)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 更新长期记忆    │
│ (持久后台任务)  │
└────────┬────────┘
         │
         ├─► 读取用户/会话的 memory_checkpoint，只处理新增消息
         ├─► 正则提取事实 + 分批 LLM 提取 + 话题检测
         ├─► 同一数据库事务保存事实、累计计数与处理进度
         └─► 更新 Redis 缓存
```

同步、流式聊天先原子追加完整回答及记忆任务，再返回完成。后台任务和手动 `/memory/update` 都使用会话游标；持久任务额外绑定稳定的 `session_ref_id`，删除后重建同名会话不会继承旧任务身份。重复处理同一历史不会重复提取或累计；相同内容在新轮次出现仍计数。旧会话首次处理按批读取已有历史。

摘要及记忆 checkpoint 仍保存在既有 JSON 字段；部署用 `create_all()` 新建 `chat_turns` 和 `memory_update_jobs`，不重写旧会话/消息。单进程内按用户串行更新记忆，LLM 调用期间不持数据库事务。后台失败不推进 checkpoint，也不把已成功回答变为 SSE 错误。历史版本的错误统计不会自动改写。

---

## 五、数据模型关系

```
users
  │
  ├──► chat_sessions ──► chat_messages
  │                         (会话消息 - 短期记忆)
  │
  └──► user_memories
        (用户记忆 - 长期记忆)
```

---

## 六、缓存策略

| 数据 | Redis Key | TTL | 说明 |
|------|------------|-----|------|
| 会话消息 | 旧 `chat_messages:{user_id}:{session_id}` | 不再回填 | PostgreSQL 权威读取，提交后仅尝试清理旧键 |
| 会话列表 | 旧 `chat_sessions:{user_id}` | 不再回填 | PostgreSQL 权威读取 |
| 用户记忆 | `user_memory:{user_id}` | 7 天 | 长期缓存 |

---

## 七、简历描述素材

### 短期记忆
- **多会话管理**：支持用户创建多个独立会话，每个会话独立存储上下文
- **追加与一致性**：PostgreSQL 原子追加、请求去重、保留历史引用与时间戳
- **增量摘要**：按 token 触发，持久化处理进度，只合并新淘汰的历史

### 长期记忆
- **双层提取**：正则快速提取 + LLM 辅助提取复杂事实
- **话题检测**：基于关键词匹配自动识别用户关注领域
- **会话摘要**：聊天使用会话级滚动摘要；每日摘要辅助方法仍保留，未接入自动聊天流程

---

## 八、持久任务运维与回滚

任务保存在 PostgreSQL `memory_update_jobs`，不是上传任务 SQLite 库。查询使用实际配置的数据库；以下是只读 SQL，可通过现有数据库管理工具执行：

```sql
SELECT id, session_ref_id, target_message_count, request_id, status,
       attempts, next_attempt_at, last_error_code
FROM memory_update_jobs ORDER BY id;
SELECT id, session_id, metadata_json->'memory_checkpoint' AS checkpoint
FROM chat_sessions ORDER BY id;
```

每次 claim 将 `attempts` 加一，最多 5 次；失败后按 5、30、120、300 秒退避，`next_attempt_at` 决定可领时间。`failed` 会阻止同会话后续目标越过失败任务，其他会话仍可处理。启动时独占所有者把遗留 `running` 恢复为 `pending`，保留尝试次数；checkpoint 防止重复提取已提交事实。

已确认故障消除后，可在项目目录用相同数据库配置执行 failed-only 重试（替换实际任务 ID）：

```powershell
uv run python -c "from backend.memory_jobs import retry_failed_job; print(retry_failed_job(123))"
```

返回 `True` 仅表示该 `failed` 任务已改为 `pending`、attempts 清零并立即可领，不表示提取完成；running/completed/不存在的 ID 返回 `False`。重试不会回滚或删除已成功 checkpoint。不要用另一进程启动 `MemoryWorker.start()`，它包含运行中任务恢复，必须由独占后端所有者执行。

没有单独的 worker 暂停环境开关或管理 HTTP API。当前可用暂停方式是正常停止后端：lifespan 调用 `MemoryWorker.aclose()`，停止领新任务并等待当前真实提取线程结束，未执行任务留在库中；重新启动会继续处理。嵌入式调用者若直接持有 worker 实例，也可 `await worker.aclose()`。协作式 deadline 不能硬杀未响应线程，退出可能需要等待。

回滚路由时关闭 `AGENT_DIRECT_CHAT_ENABLED` 并清空 `AGENT_ROUTER_MODEL`。回滚存储实现前正常停机、备份数据库并保留新任务表，禁止直接启动仍全量重写历史的旧聊天代码。`replace_all_messages` 维护前必须停 worker，人工核对旧任务、memory/summary checkpoint 和请求重放记录；该维护函数不会替你协调这些状态。

*文档更新时间：2026-10-03*
