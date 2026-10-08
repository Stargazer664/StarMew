"""Durable actual tool outcomes; never insert tools into conversational history."""
from dataclasses import asdict
from datetime import datetime

from .models import ChatTurn, ChatSession, ToolExecution
from .run_control import current_control, check_cancelled, RunCancelled


def begin_tool(tool_call_id, name, arguments):
    control = current_control()
    if control is None or tool_call_id is None:
        return None
    check_cancelled()
    identity = control.identity
    with control.session_factory() as db:
        turn = db.query(ChatTurn).join(ChatSession).filter(ChatTurn.id == identity.turn_id,
            ChatTurn.session_ref_id == identity.session_ref_id, ChatTurn.request_id == identity.request_id,
            ChatSession.instance_id == identity.session_instance_id).first()
        if turn is None or turn.status != 'running':
            raise RunCancelled('REQUEST_CANCELLED')
        row = ToolExecution(turn_id=turn.id, tool_call_id=tool_call_id,
                            tool_name=name, arguments=arguments)
        db.add(row)
        db.commit()
        return row.id


def finish_tool(record_id, *, result=None, status=None):
    control = current_control()
    if control is None or record_id is None:
        return
    with control.session_factory() as db:
        row = db.query(ToolExecution).join(ChatTurn).join(ChatSession).filter(ToolExecution.id == record_id,
            ChatTurn.id == control.identity.turn_id,
            ChatTurn.session_ref_id == control.identity.session_ref_id,
            ChatTurn.request_id == control.identity.request_id,
            ChatSession.instance_id == control.identity.session_instance_id).first()
        if row is None:
            return
        row.status = status or {'ok': 'completed', 'empty': 'completed', 'error': 'failed', 'skipped': 'skipped'}[result.status]
        row.result = asdict(result) if result is not None else None
        row.cancel_requested = control.cancelled.is_set()
        row.finished_at = datetime.utcnow()
        db.commit()
