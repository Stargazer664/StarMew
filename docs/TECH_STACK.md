# StarMew 技术栈与项目详解

---

## 一、技术栈总览

### 后端框架
| 技术 | 用途 | 版本 |
|------|------|------|
| **Python** | 开发语言 | 3.10+ |
| **FastAPI** | Web 框架 | 0.115+ |
| **Uvicorn** | ASGI 服务器 | 0.30+ |

### AI / LLM
| 技术 | 用途 | 版本 |
|------|------|------|
| **LangChain** | LLM 应用框架 | 0.3+ |
| **LangGraph** | Agent 状态机 | 0.2+ |
| **火山引擎 API** | LLM 调用 | Doubao 系列 |

### 数据库与缓存
| 技术 | 用途 | 版本 |
|------|------|------|
| **PostgreSQL** | 关系型数据库 | 16 |
| **Redis** | 缓存/会话存储 | 7 |
| **SQLAlchemy** | ORM | 2.0+ |

### 向量与检索
| 技术 | 用途 | 版本 |
|------|------|------|
| **Milvus** | 向量数据库 | 2.5.14 |
| **BGE-m3** | 稠密向量模型 | - |
| **BM25** | 稀疏向量检索 | 自研 |
| **Jina Reranker** | 重排模型 | v3 |

### 文档处理
| 技术 | 用途 |
|------|------|
| **PyPDFLoader** | PDF 解析 |
| **Docx2txtLoader** | Word 解析 |
| **UnstructuredExcelLoader** | Excel 解析 |
| **RecursiveCharacterTextSplitter** | 文本分块 |

### 认证与安全
| 技术 | 用途 |
|------|------|
| **JWT** | 身份令牌 |
| **PBKDF2** | 密码哈希 |
| **OAuth2PasswordBearer** | 认证方案 |

### DevOps
| 技术 | 用途 |
|------|------|
| **Docker Compose** | 容器编排 |
| **uv** | Python 包管理 |

---

## 二、项目架构图

```
┌─────────────────────────────────────────────────────────────┐
│                     Vue 3 前端 (CDN)                        │
│               http://localhost:8000                         │
└─────────────────────────────────────────────────────────────┘
                              │
                    HTTP / SSE 流式响应
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI 后端                           │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────┐    │
│  │  认证   │  │  聊天   │  │  文档   │  │   评估      │    │
│  │ /auth   │  │ /chat   │  │/documents│ │ /evaluation │    │
│  └─────────┘  └─────────┘  └─────────┘  └─────────────┘    │
└─────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  LangGraph      │  │   PostgreSQL    │  │     Redis       │
│  ReAct Agent    │  │   (持久化)      │  │    (缓存)       │
└─────────────────┘  └─────────────────┘  └─────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│                    RAG Pipeline                             │
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │   混合检索    │ →  │  LLM 打分    │ →  │  查询重写    │   │
│  │ (Dense+Sparse)│    │ (yes/no)     │    │(Step/HyDE)   │   │
│  └──────────────┘    └──────────────┘    └──────────────┘   │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│                      向量存储层                              │
│                                                             │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────┐ │
│  │ Milvus  │  │  BM25   │  │ L3 块   │  │ L1/L2 父块      │ │
│  │(混合检索)│  │ (稀疏)  │  │(叶子块) │  │ (PostgreSQL)   │ │
│  └─────────┘  └─────────┘  └─────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## 三、核心模块详解

### 1. Agent 模块 (`backend/agent.py`)

**ReAct Agent 架构**
```
Thought → Action → Observation → ...
```

- **StateGraph**：使用 LangGraph 定义 Agent 状态机
- **reason 节点**：模型推理决定是否使用工具
- **act 节点**：执行工具并返回 ToolMessage
- **流式输出**：SSE 实时推送 Agent 思考过程

**对话管理**
```python
class ConversationStorage:
    - save()      # 保存到 PostgreSQL + Redis
    - load()      # 优先 Redis，失败查数据库
    - list_sessions()  # 会话列表
    - delete_session() # 删除会话
```

### 2. RAG Pipeline (`backend/rag_pipeline.py`)

**工作流**
```
用户问题
    │
    ▼
┌───────────────────┐
│ retrieve_initial   │ 混合检索 (Dense + Sparse)
│   (检索初始文档)   │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ grade_documents   │ LLM 打分 (yes/no)
│   (相关性评估)    │
└─────────┬─────────┘
          │
    ┌─────┴─────┐
    │           │
   yes         no
    │           │
    ▼           ▼
  [END]   ┌───────────────────┐
          │ rewrite_question  │ 查询重写
          │  (策略选择)        │
          └─────────┬─────────┘
                    │
                    ▼
          ┌───────────────────┐
          │ retrieve_expanded │ 扩展检索
          │  (重写后检索)     │
          └─────────┬─────────┘
                    │
                    ▼
                  [END]
```

### 3. 混合检索 (`backend/milvus_client.py` + `embedding.py`)

**检索流程**
```
查询文本
    │
    ├──→ BGE-m3 (Dense) ──→ 稠密向量
    │
    └──→ BM25 (Sparse) ──→ 稀疏向量
              │
              ▼
        RRF 融合 (k=60)
              │
              ▼
        Jina Rerank 重排
              │
              ▼
        Auto-merging 合并
              │
              ▼
           Top-K 结果
```

### 4. 三级分块 (`backend/document_loader.py`)

```
原始文档
    │
    ▼
┌─────────────────────────────────────┐
│  L1 (1200 tokens, overlap=240)      │ ← 父块
│  ┌─────────────────────────────────┐│
│  │  L2 (600 tokens, overlap=120)   ││ ← 中间块
│  │  ┌─────────────────────────────┐││
│  │  │ L3 (300 tokens, overlap=60) │││ ← 叶子块
│  │  │        (写入 Milvus)        │││
│  │  └─────────────────────────────┘││
│  └─────────────────────────────────┘│
└─────────────────────────────────────┘
```

**Auto-merging**：L3 召回后，按 root_chunk_id 分组，>=2 个相同父块则合并为 L1

### 5. 查询扩展策略

| 策略 | 触发条件 | 实现 |
|------|----------|------|
| **Step-back** | 细节问题 | 生成抽象概念问题 → 检索 → 回答 → 扩展 |
| **HyDE** | 模糊问题 | 生成假设性文档 → 向量检索 |
| **Complex** | 复合问题 | Step-back + HyDE 组合 |

### 6. 认证授权 (`backend/auth.py`)

```
用户注册 ──→ PBKDF2 哈希 ──→ PostgreSQL
                        │
                        ▼
用户登录 ──→ 验证哈希 ──→ JWT 签发 ──→ 前端存储
                        │
                        ▼
                  Bearer Token ──→ 受保护路由
```

**权限体系**
- `user`：聊天、查看/删除自己会话
- `admin`：文档上传/删除、评估

---

## 四、API 路由

| 路由 | 方法 | 权限 | 功能 |
|------|------|------|------|
| `/auth/register` | POST | 公开 | 用户注册 |
| `/auth/login` | POST | 公开 | 用户登录 |
| `/auth/me` | GET | 用户 | 获取当前用户 |
| `/chat` | POST | 用户 | 非流式聊天 |
| `/chat/stream` | POST | 用户 | 流式聊天 (SSE) |
| `/sessions` | GET | 用户 | 会话列表 |
| `/sessions/{id}` | GET/DELETE | 用户 | 会话消息/删除 |
| `/documents` | GET | 管理员 | 文档列表 |
| `/documents/upload` | POST | 管理员 | 上传文档 |
| `/documents/{name}` | DELETE | 管理员 | 删除文档 |
| `/memory` | GET | 用户 | 获取记忆 |
| `/memory/update` | POST | 用户 | 更新记忆 |
| `/memory/reset` | DELETE | 用户 | 重置记忆 |
| `/evaluation/run` | POST | 用户 | 运行评估 |

---

## 五、数据模型

### User
```
id, username, password_hash, role, created_at
```

### ChatSession
```
id, user_id, session_id, metadata_json, updated_at, created_at
```

### ChatMessage
```
id, session_ref_id, message_type, content, timestamp, rag_trace
```

### ParentChunk
```
chunk_id(PK), text, filename, file_type, page_number,
parent_chunk_id, root_chunk_id, chunk_level, chunk_idx
```

### UserMemory
```
id, user_id, preferences, facts, session_summaries,
detected_topics, total_sessions, total_messages
```

---

## 六、简历项目描述模板

### 中文版

**项目名称**：StarMew - 智能 RAG Agent 系统

**项目描述**：
基于检索增强生成（RAG）的智能 Agent 系统，融合混合向量搜索、三级分块策略与 ReAct 推理框架，支持流式输出和用户记忆管理。

**技术亮点**：
- 设计并实现混合检索架构，结合 BGE-m3 稠密向量与自研 BM25 稀疏向量，通过 RRF 融合算法提升检索效果
- 设计三级滑动窗口分块策略，配合 Auto-merging 合并机制，减少检索碎片化
- 基于 LangGraph 构建 ReAct Agent，支持工具调用、流式 SSE 输出、跨线程 RAG 步骤推送
- 实现 RAG Pipeline 工作流，包含 LLM 相关性打分、Step-back/HyDE 查询扩展
- 设计 JWT + PBKDF2 认证体系，支持用户和管理员两级权限

**技术栈**：Python, FastAPI, LangChain, LangGraph, PostgreSQL, Redis, Milvus, BGE-m3

---

### 英文版

**Project: StarMew - Intelligent RAG Agent System**

**Description:**
Built a Retrieval-Augmented Generation (RAG) system with hybrid vector search, 3-level chunking, and ReAct agent framework featuring streaming output and user memory management.

**Key Achievements:**
- Designed hybrid retrieval combining BGE-m3 dense vectors with custom BM25 sparse vectors using RRF fusion
- Implemented 3-level sliding window chunking with Auto-merging to reduce fragmentation
- Built ReAct Agent with LangGraph StateGraph supporting tool calling, SSE streaming, and cross-thread RAG step dispatching
- Developed RAG Pipeline with LLM grading, Step-back/HyDE query expansion strategies
- Implemented JWT + PBKDF2 authentication with user/admin RBAC

**Tech Stack:** Python, FastAPI, LangChain, LangGraph, PostgreSQL, Redis, Milvus, BGE-m3, Jina Reranker

---

## 七、简历关键词清单

```
# 框架
FastAPI, LangChain, LangGraph, Uvicorn, SQLAlchemy

# AI / LLM
RAG, ReAct Agent, LLM, Embedding, Vector Search, Rerank

# 数据库
PostgreSQL, Redis, Milvus

# 检索技术
Hybrid Search, Dense/Sparse Retrieval, BM25, RRF, HNSW

# 分块策略
Chunking, Auto-merging, Parent-Child, Sliding Window

# 查询优化
Step-back, HyDE, Query Expansion, Query Rewrite

# 其他
JWT, SSE, Docker, API Design, RBAC
```

---

*文档更新时间：2026-05-14*