"""One owner for a foreground turn's calls, signatures and monotonic deadline."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import json
import math
import os
import time
import unicodedata
from .run_control import check_cancelled
from typing import Literal

StopReason = Literal['decision_limit', 'tool_limit', 'duplicate_call', 'deadline', 'evidence_sufficient']


class AgentDeadline(TimeoutError):
    def __init__(self):
        super().__init__('本轮处理时间已耗尽，回答未完成，请稍后重试。')


@dataclass(frozen=True)
class AgentBudget:
    max_decisions: int = 4
    max_tool_calls: int = 4
    max_kb_calls: int = 2
    max_retrieval_calls: int = 4
    total_timeout_seconds: float = 120
    final_reserve_seconds: float = 20

    @classmethod
    def from_env(cls):
        values = {}
        for name, default in cls().__dict__.items():
            raw = os.getenv('AGENT_' + name.upper(), str(default))
            try:
                value = int(raw) if name.startswith('max_') else float(raw)
                if not math.isfinite(value) or value <= 0:
                    raise ValueError
            except (ValueError, OverflowError):
                raise ValueError('Invalid AGENT_' + name.upper()) from None
            values[name] = value
        if values['final_reserve_seconds'] >= values['total_timeout_seconds']:
            raise ValueError('AGENT_FINAL_RESERVE_SECONDS must be below total timeout')
        return cls(**values)


def call_signature(name, args):
    def normalize(value):
        if isinstance(value, str):
            return ' '.join(unicodedata.normalize('NFKC', value).split())
        if isinstance(value, dict):
            return {key: normalize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [normalize(item) for item in value]
        return value
    return json.dumps({'tool_name': name, 'normalized_args': normalize(args)},
                      ensure_ascii=False, sort_keys=True, separators=(',', ':'))


@dataclass
class AgentRunState:
    budget: AgentBudget = field(default_factory=AgentBudget.from_env)
    clock: object = field(default_factory=lambda: time.monotonic, repr=False)
    started_at: float = field(init=False)
    decision_count: int = 0
    tool_count: int = 0
    retrieval_count: int = 0
    kb_signatures: set = field(default_factory=set)
    signatures: dict = field(default_factory=dict)
    stop_reason: StopReason | None = None
    answer_status: str = 'failed'

    def __post_init__(self):
        self.started_at = self.clock()

    def remaining(self, *, final=False):
        reserve = 0 if final else self.budget.final_reserve_seconds
        return max(0.0, self.budget.total_timeout_seconds - (self.clock() - self.started_at) - reserve)

    def timeout(self, cap=None, *, final=False):
        check_cancelled()
        remaining = self.remaining(final=final)
        if remaining <= 0:
            self.stop_reason = 'deadline'
            raise AgentDeadline()
        return min(cap, remaining) if cap is not None else remaining

    def check(self, *, decision=False):
        check_cancelled()
        if self.remaining() <= 0:
            self.stop_reason = 'deadline'
        elif self.stop_reason is None and self.tool_count >= self.budget.max_tool_calls:
            self.stop_reason = 'tool_limit'
        elif self.stop_reason is None and decision and self.decision_count >= self.budget.max_decisions:
            self.stop_reason = 'decision_limit'
        return self.stop_reason

    def claim_tool(self, name, args):
        if self.check():
            return None, 'CALL_LIMIT_REACHED'
        signature = call_signature(name, args)
        prior = self.signatures.get(signature)
        if prior and not (prior['attempts'] == 1 and prior['retryable']):
            self.stop_reason = 'duplicate_call'
            return None, 'DUPLICATE_CALL'
        if name == 'search_knowledge_base' and signature not in self.kb_signatures:
            if len(self.kb_signatures) >= self.budget.max_kb_calls:
                self.stop_reason = 'tool_limit'
                return None, 'CALL_LIMIT_REACHED'
            self.kb_signatures.add(signature)
        self.tool_count += 1
        self.signatures[signature] = {'attempts': (prior or {}).get('attempts', 0) + 1, 'retryable': False}
        return signature, None

    def record_tool(self, signature, result):
        self.signatures[signature]['retryable'] = result.status == 'error' and result.retryable
        self.check()

    def claim_retrieval(self):
        self.timeout()
        if self.retrieval_count >= self.budget.max_retrieval_calls:
            self.stop_reason = 'tool_limit'
            return False
        self.retrieval_count += 1
        return True

    @property
    def recursion_limit(self):
        return max(32, self.budget.max_decisions * 2 + 5)


_run_state = ContextVar('agent_run_state', default=None)


def current_run_state():
    return _run_state.get()


@contextmanager
def use_run_state(state):
    token = _run_state.set(state)
    try:
        yield state
    finally:
        _run_state.reset(token)


def remaining_timeout(cap=None, *, final=False):
    state = current_run_state()
    return state.timeout(cap, final=final) if state else cap


def model_timeout_kwargs(purpose):
    timeout = remaining_timeout(final=purpose == 'final_answer')
    return {'timeout': timeout} if timeout is not None else {}


def network_timeout_kwargs():
    """Preserve standalone SDK defaults outside an admitted foreground turn."""
    timeout = remaining_timeout()
    return {'timeout': timeout} if timeout is not None else {}
