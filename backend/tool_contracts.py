"""Request-owned, serializable results shared by all external tools."""
from dataclasses import dataclass, field, fields
import json
import logging
from typing import Literal

from pydantic import ValidationError
import requests

ToolStatus = Literal['ok', 'empty', 'error', 'skipped']
ToolErrorCode = Literal['NOT_CONFIGURED', 'INVALID_ARGUMENT', 'UPSTREAM_TIMEOUT',
                        'UPSTREAM_ERROR', 'UNKNOWN_TOOL', 'CALL_LIMIT_REACHED', 'DUPLICATE_CALL']


@dataclass
class ToolResult:
    status: ToolStatus
    content: str
    code: ToolErrorCode | None = None
    retryable: bool = False
    evidence: list[dict] = field(default_factory=list)
    evidence_status: Literal['sufficient', 'insufficient', 'unknown'] | None = None

    def to_json(self, *, include_evidence: bool = True) -> str:
        # Model content already contains formatted evidence. Retain the full
        # result in ToolMessage.artifact without duplicating its text in tokens.
        payload = {item.name: getattr(self, item.name) for item in fields(self)
                   if include_evidence or item.name != 'evidence'}
        return json.dumps(payload, ensure_ascii=False)


_ERROR_TEXT = {
    'NOT_CONFIGURED': '工具服务未配置，无法执行此操作。',
    'INVALID_ARGUMENT': '工具参数无效，请提供完整且有效的参数。',
    'UPSTREAM_TIMEOUT': '工具服务请求超时，尚未获取结果。',
    'UPSTREAM_ERROR': '工具服务执行失败，尚未获取结果。',
    'UNKNOWN_TOOL': '请求的工具不存在。',
    'CALL_LIMIT_REACHED': '本轮工具调用预算已用尽，请使用已有结果。',
    'DUPLICATE_CALL': '该工具调用已执行，请使用已有结果。',
}


def tool_error(code: ToolErrorCode, *, retryable: bool = False) -> ToolResult:
    return ToolResult('skipped' if code in ('CALL_LIMIT_REACHED', 'DUPLICATE_CALL') else 'error',
                      _ERROR_TEXT[code], code=code, retryable=retryable)


def tool_exception(exc: Exception) -> ToolResult:
    """Keep exception text/stack in server logs; return only safe fixed errors."""
    logging.getLogger(__name__).error('Tool operation failed', exc_info=exc)
    if isinstance(exc, ValidationError):
        return tool_error('INVALID_ARGUMENT')
    if isinstance(exc, (TimeoutError, requests.exceptions.Timeout)):
        return tool_error('UPSTREAM_TIMEOUT', retryable=True)
    retryable = isinstance(exc, requests.exceptions.ConnectionError)
    if isinstance(exc, requests.exceptions.HTTPError):
        status = getattr(exc.response, 'status_code', None)
        retryable = status == 429 or (isinstance(status, int) and status >= 500)
    return tool_error('UPSTREAM_ERROR', retryable=retryable)
