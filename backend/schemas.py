from pydantic import BaseModel, Field, model_validator
from uuid import UUID
from typing import Optional, List


class RegisterRequest(BaseModel):
    username: str
    password: str
    role: Optional[str] = "user"
    admin_code: Optional[str] = None


class LoginRequest(BaseModel):
    username: str
    password: str


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str
    role: str


class CurrentUserResponse(BaseModel):
    username: str
    role: str


class ChatRequest(BaseModel):
    message: str
    request_id: Optional[UUID] = None
    session_id: Optional[str] = "default_session"


class ChatCancelRequest(BaseModel):
    session_id: str = 'default_session'
    supplement_message: Optional[str] = Field(default=None, min_length=1, max_length=32000)
    supplement_request_id: Optional[UUID] = None

    @model_validator(mode='after')
    def validate_supplement(self):
        if (self.supplement_message is None) != (self.supplement_request_id is None):
            raise ValueError('补充消息和请求 ID 必须同时提供')
        if self.supplement_message is not None and not self.supplement_message.strip():
            raise ValueError('补充消息不能为空')
        return self


def public_rag_trace(trace):
    """One projection for HTTP, persisted-history reads, and SSE (including replay)."""
    if trace is None:
        return None
    selected = 'answer_evidence' in trace
    hidden = {'query', 'internal_query', 'expanded_query', 'rewrite_query', 'step_back_question',
              'step_back_answer', 'hypothetical_doc', 'rerank_error'}
    if selected:
        hidden |= {'retrieved_chunks', 'initial_retrieved_chunks', 'expanded_retrieved_chunks', 'context_chunks'}
    def project(value):
        if isinstance(value, dict):
            return {key: project(item) for key, item in value.items() if key not in hidden}
        if isinstance(value, list):
            return [project(item) for item in value]
        return value
    result = project(trace)
    if selected:
        evidence = result['answer_evidence']
        result['retrieved_chunks'] = [dict(item, filename=item['source'],
            page_number=int(str(item['page'])) if str(item['page']).isdigit() else item['page']) for item in evidence]
        result.setdefault('tool_used', bool(evidence or trace.get('evidence_omitted_count')))
        result.setdefault('tool_name', None)
    return result


class AnswerEvidence(BaseModel):
    citation_id: str
    source: str
    page: str | int
    text: str
    rank: int
    score: Optional[float] = None


class RetrievedChunk(BaseModel):
    filename: str
    page_number: Optional[str | int] = None
    text: Optional[str] = None
    score: Optional[float] = None
    rrf_rank: Optional[int] = None
    rerank_score: Optional[float] = None
    citation_id: Optional[str] = None
    source: Optional[str] = None
    page: Optional[str | int] = None
    rank: Optional[int] = None


class RagTrace(BaseModel):
    tool_used: bool = False
    tool_name: Optional[str] = None
    answer_evidence: Optional[List[AnswerEvidence]] = None
    evidence_omitted_count: Optional[int] = None

    @model_validator(mode='before')
    @classmethod
    def project_public_trace(cls, value):
        return public_rag_trace(value) if isinstance(value, dict) else value
    query: Optional[str] = None
    expanded_query: Optional[str] = None
    step_back_question: Optional[str] = None
    step_back_answer: Optional[str] = None
    expansion_type: Optional[str] = None
    hypothetical_doc: Optional[str] = None
    retrieval_stage: Optional[str] = None
    grade_score: Optional[str] = None
    grade_route: Optional[str] = None
    rewrite_needed: Optional[bool] = None
    rewrite_strategy: Optional[str] = None
    rewrite_query: Optional[str] = None
    rerank_enabled: Optional[bool] = None
    rerank_applied: Optional[bool] = None
    rerank_model: Optional[str] = None
    rerank_endpoint: Optional[str] = None
    rerank_error: Optional[str] = None
    retrieval_mode: Optional[str] = None
    candidate_k: Optional[int] = None
    leaf_retrieve_level: Optional[int] = None
    auto_merge_enabled: Optional[bool] = None
    auto_merge_applied: Optional[bool] = None
    auto_merge_threshold: Optional[int] = None
    auto_merge_replaced_chunks: Optional[int] = None
    auto_merge_steps: Optional[int] = None
    retrieved_chunks: Optional[List[RetrievedChunk]] = None
    initial_retrieved_chunks: Optional[List[RetrievedChunk]] = None
    expanded_retrieved_chunks: Optional[List[RetrievedChunk]] = None
    evidence_status: Optional[str] = None
    retrieval_status: Optional[str] = None
    retrieval_records: Optional[List[dict]] = None
    relevance_filters: Optional[List[dict]] = None
    retrieval_rounds: Optional[List[dict]] = None
    grading_records: Optional[List[dict]] = None
    expansion_status: Optional[str] = None
    expansion_stage: Optional[str] = None
    expansion_error_code: Optional[str] = None
    expansion_retryable: Optional[bool] = None


class ChatResponse(BaseModel):
    response: str
    rag_trace: Optional[RagTrace] = None


class MessageInfo(BaseModel):
    type: str
    content: str
    timestamp: str
    status: str = 'complete'
    rag_trace: Optional[RagTrace] = None


class SessionMessagesResponse(BaseModel):
    messages: List[MessageInfo]
    pending_run: Optional[dict] = None


class SessionInfo(BaseModel):
    session_id: str
    updated_at: str
    message_count: int


class SessionListResponse(BaseModel):
    sessions: List[SessionInfo]


class SessionDeleteResponse(BaseModel):
    session_id: str
    message: str


class DocumentInfo(BaseModel):
    filename: str
    file_type: str
    chunk_count: int
    uploaded_at: Optional[str] = None


class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]


class DocumentUploadResponse(BaseModel):
    filename: str
    chunks_processed: int
    message: str


class DocumentUploadStartResponse(BaseModel):
    job_id: str
    filename: str
    message: str


class UploadStepInfo(BaseModel):
    key: str
    label: str
    percent: int
    status: str
    message: str = ""


class DocumentUploadJobResponse(BaseModel):
    job_id: str
    filename: str
    status: str
    current_step: str
    message: str
    total_chunks: int = 0
    processed_chunks: int = 0
    error: Optional[str] = None
    created_at: str
    updated_at: str
    steps: List[UploadStepInfo]
    retryable: bool = False
    recovery_required: bool = False
    attempt: int = 1


class DocumentDeleteStartResponse(BaseModel):
    job_id: str
    filename: str
    message: str


class DocumentDeleteJobResponse(DocumentUploadJobResponse):
    pass


class DocumentDeleteResponse(BaseModel):
    filename: str
    chunks_deleted: int
    message: str


class MemoryEntryResponse(BaseModel):
    id: int
    key: str
    category: str
    value: str
    scope: str
    project: str
    status: str
    source_session_id: Optional[int] = None
    source_message_id: Optional[int] = None
    source_quote: str = ""
    source_kind: str
    created_at: str
    expires_at: Optional[str] = None


class MemoryResponse(BaseModel):
    """用户长期记忆响应"""
    username: str
    preferences: dict = {}
    facts: dict = {}
    session_summaries: dict = {}
    detected_topics: list = []
    stats: dict = {}
    entries: list[MemoryEntryResponse] = Field(default_factory=list)


class MemoryUpdateResponse(BaseModel):
    """记忆更新响应"""
    message: str
    updated_facts: dict = {}
    updated_preferences: dict = {}


class MemoryResetResponse(BaseModel):
    """记忆重置响应"""
    message: str
    username: str


# ============ RAG 评估 Schema ============

class EvaluationRequest(BaseModel):
    """评估请求"""
    question: str
    answer: str
    retrieved_docs: list = []
    ground_truth: Optional[str] = None
    rag_trace: Optional[dict] = None


class EvaluationMetrics(BaseModel):
    """评估指标得分"""
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    factual_correctness: Optional[float] = None
    answer_correctness: Optional[float] = Field(
        default=None, deprecated=True,
        description="独立的旧 AnswerCorrectness 指标；不作为 FactualCorrectness 的别名。",
    )


class EvaluationResult(BaseModel):
    """评估结果"""
    timestamp: str = ""
    question: str = ""
    answer: str = ""
    answer_length: int = 0
    contexts_count: int = 0
    metrics: EvaluationMetrics = EvaluationMetrics()
    overall_score: Optional[float] = None
    evaluation_status: str = "unknown"
    metric_status: dict[str, str] = Field(default_factory=dict)
    metric_errors: dict[str, str] = Field(default_factory=dict)
    ragas_native: Optional[bool] = None
    ragas_version: str = ""
    ragas_style: bool = True
    detailed: bool = False
    retrieval_mode: Optional[str] = None
    rerank_applied: bool = False
    auto_merge_applied: bool = False
    latency: Optional[float] = None
    error: Optional[str] = None


class EvaluationResponse(BaseModel):
    """评估响应"""
    evaluation: EvaluationResult
    average_scores: dict


class EvaluationHistoryResponse(BaseModel):
    """评估历史响应"""
    evaluations: list[EvaluationResult]
    average_scores: dict
    total: int


class BatchEvaluationRequest(BaseModel):
    """批量评测请求"""
    dataset_path: Optional[str] = None
    questions: Optional[list[str]] = None
    max_concurrency: int = 3
    detailed: bool = False


class BatchEvaluationResponse(BaseModel):
    """批量评测响应"""
    result_count: int
    results: list[EvaluationResult]
    average_scores: dict
    export_path: Optional[str] = None


class EvalSampleSchema(BaseModel):
    """评测样本"""
    question: str
    ground_truth: Optional[str] = None
    expected_docs: Optional[list[str]] = None
    metadata: dict = {}


class EvalDatasetResponse(BaseModel):
    """评测数据集响应"""
    total_samples: int
    samples: list[EvalSampleSchema]
