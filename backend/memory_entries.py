"""Memory validation, versioning and selection; transactions belong to the caller."""
from datetime import datetime, timezone
import json
import re

from .models import MemoryEntry


def utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def serialize(row):
    fields = ('id', 'key', 'category', 'value', 'scope', 'project', 'status',
              'source_session_id', 'source_message_id', 'source_quote', 'source_kind')
    data = {name: getattr(row, name) for name in fields}
    data['created_at'] = row.created_at.isoformat() + 'Z'
    data['expires_at'] = row.expires_at.isoformat() + 'Z' if row.expires_at else None
    if row.status == 'active' and row.expires_at and row.expires_at <= utcnow():
        data['status'] = 'expired'
    return data


def migrate_legacy(db, memory):
    """Idempotent lazy import; revoked/superseded identities prevent resurrection."""
    existing = {r.key for r in db.query(MemoryEntry).filter_by(user_id=memory.user_id, scope='global')}
    for category, values in [('fact', memory.facts), ('preference', memory.preferences)]:
        for key, value in (values or {}).items():
            if key in existing or not value:
                continue
            db.add(MemoryEntry(user_id=memory.user_id, category=category, key=key,
                value=json.dumps(value, ensure_ascii=False) if isinstance(value, list) else str(value),
                scope='global', source_kind='legacy', source_quote=''))
            existing.add(key)
    db.flush()


def validate_operations(operations, messages, session_ref_id):
    if not isinstance(operations, list) or len(operations) > 128:
        raise ValueError('Invalid memory operations')
    validated = []
    for raw in operations:
        if not isinstance(raw, dict):
            raise ValueError('Invalid memory operation')
        op = dict(raw)
        # Some local models omit the paraphrase. A verified verbatim source is
        # a safe representation; never invent a replacement value.
        if op.get('action') == 'upsert' and 'value' not in op:
            op['value'] = op.get('quote', '')
        for field, maximum in [('key', 120), ('category', 80), ('value', 2000), ('quote', 2000), ('project', 160)]:
            value = op.get(field, '' if field in {'value', 'project'} else None)
            if not isinstance(value, str) or len(value) > maximum:
                raise ValueError('Invalid memory ' + field)
            op[field] = value.strip()
        if not op['key'] or not op['category'] or not op['quote']:
            raise ValueError('Memory needs a key, category and source quote')
        if op.get('action') not in {'upsert', 'revoke'} or op.get('scope') not in {'global', 'project', 'session'}:
            raise ValueError('Invalid memory action/scope')
        index = op.get('source_index')
        if type(index) is not int or not 0 <= index < len(messages):
            raise ValueError('Invalid memory source index')
        source = messages[index]
        if source.get('type') != 'human' or op['quote'] not in source.get('content', ''):
            raise ValueError('Memory source must quote a user message')
        if op['action'] == 'upsert' and not op['value']:
            raise ValueError('Empty memory value')
        if op['scope'] == 'project':
            if not op['project'] or op['project'].casefold() not in op['quote'].casefold():
                raise ValueError('Project scope must be explicitly named in source quote')
            op['project'] = op['project'].casefold()
        else:
            op['project'] = ''
        if (op['action'] == 'upsert' and op['scope'] == 'global'
                and op['category'] not in {'fact', 'preference'}
                and re.search(r'项目|仓库|\bproject\b|\brepository\b|\bcodebase\b', op['quote'], re.I)
                and not re.search(r'(?:所有|全部|任何|每个)(?:的)?(?:项目|仓库|代码库)|\b(?:all|every|any)\s+(?:my\s+)?(?:projects?|codebases?|repositories|repository)\b', op['quote'], re.I)):
            # A misclassified project constraint must not become user-global.
            # Without a validated project name, limit it to this conversation.
            op['scope'] = 'session'
        expiration = op.get('expires_at')
        if expiration:
            try:
                parsed = datetime.fromisoformat(expiration.replace('Z', '+00:00'))
                if parsed.tzinfo:
                    parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
                op['expires_at'] = parsed
            except (ValueError, TypeError, AttributeError) as exc:
                raise ValueError('Invalid memory expiry') from exc
        else:
            op['expires_at'] = None
        # Undated plans are temporary, never silently user-global.
        if op['action'] == 'upsert' and op['category'] in {'plan', 'task', 'temporary'} and not op['expires_at']:
            op['scope'] = 'session'
            op['project'] = ''
        if op['scope'] == 'session' and session_ref_id is None:
            raise ValueError('Session-scoped memory requires a session')
        op['source_message_id'] = source.get('id')
        op['key'] = op['key'].casefold()
        validated.append(op)
    return sorted(validated, key=lambda op: op['source_index'])


def apply_operations(db, user_id, operations, session_ref_id):
    for op in operations:
        session_scope = session_ref_id if op['scope'] == 'session' else None
        prior = db.query(MemoryEntry).filter_by(user_id=user_id, key=op['key'],
            scope=op['scope'], project=op['project'], scope_session_id=session_scope).order_by(MemoryEntry.id.desc()).all()
        source_id = op.get('source_message_id')
        # Explicit forgetting and later declarations win over old queued sources.
        if source_id is not None and any(max(r.source_message_id or 0, r.forgotten_through_message_id or 0) >= source_id for r in prior):
            continue
        for row in prior:
            if row.status == 'active':
                row.status = 'superseded' if op['action'] == 'upsert' else 'revoked'
        db.add(MemoryEntry(user_id=user_id, key=op['key'], category=op['category'],
            value=op['value'], scope=op['scope'], project=op['project'],
            scope_session_id=session_scope, source_session_id=session_ref_id,
            source_message_id=source_id, source_quote=op['quote'],
            expires_at=op['expires_at'], status='active' if op['action'] == 'upsert' else 'revoked'))
        db.flush()


def active_rows(db, user_id):
    return [r for r in db.query(MemoryEntry).filter_by(user_id=user_id, status='active').all()
            if not r.expires_at or r.expires_at > utcnow()]


def project_legacy(db, memory):
    facts, preferences = {}, {}
    for row in active_rows(db, memory.user_id):
        if row.scope != 'global':
            continue
        value = row.value
        if row.key == 'allergies':
            try:
                value = json.loads(value)
            except ValueError:
                value = [value]
        if row.category == 'preference':
            preferences[row.key] = value
        else:
            facts[row.key] = value
    memory.facts, memory.preferences = facts, preferences


def select_context(rows, *, query='', session_ref_id=None):
    text = query.casefold()
    selected = [r for r in rows if r.scope == 'global'
        or (r.scope == 'session' and r.scope_session_id == session_ref_id and session_ref_id is not None)
        or (r.scope == 'project' and ((r.project and r.project in text)
            or (session_ref_id is not None and r.source_session_id == session_ref_id)))]
    terms = set(re.findall(r'[a-z0-9_]+|[\u4e00-\u9fff]', text))
    def rank(row):
        haystack = (row.key + ' ' + row.value + ' ' + row.project).casefold()
        return (sum(t in haystack for t in terms), row.created_at, row.id)
    return sorted(selected, key=rank, reverse=True)
