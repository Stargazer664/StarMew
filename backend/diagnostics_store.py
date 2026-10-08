"""Local diagnostics projections. Business journals remain the source of truth."""
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
import json
import math
from pathlib import Path
import sqlite3


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class DiagnosticsStore:
    def __init__(self, path):
        self.path = Path(path).resolve()

    @contextmanager
    def connection(self, write=False):
        # Read endpoints must neither create the database nor initialize tables.
        connection = sqlite3.connect(str(self.path) if write else self.path.as_uri() + '?mode=ro',
                                     uri=not write, timeout=0.25)
        connection.row_factory = sqlite3.Row
        try:
            if write:
                connection.execute('BEGIN IMMEDIATE')
            yield connection
            if write:
                connection.commit()
        except BaseException:
            if write:
                connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection(write=True) as db:
            for table in ('traces', 'spans'):
                db.execute(f'CREATE TABLE IF NOT EXISTS diagnostic_{table} ('
                           'id TEXT PRIMARY KEY, trace_id TEXT, scope TEXT, job_id TEXT, operation TEXT, '
                           'status TEXT, started_at TEXT, payload TEXT NOT NULL)')
                db.execute(f'CREATE INDEX IF NOT EXISTS diagnostic_{table}_trace ON diagnostic_{table}(trace_id)')
                db.execute(f'CREATE INDEX IF NOT EXISTS diagnostic_{table}_job ON diagnostic_{table}(operation,job_id)')
                db.execute(f'CREATE INDEX IF NOT EXISTS diagnostic_{table}_time ON diagnostic_{table}(started_at)')
            db.execute('CREATE TABLE IF NOT EXISTS diagnostic_events ('
                       'id INTEGER PRIMARY KEY AUTOINCREMENT, event_key TEXT UNIQUE, trace_id TEXT, '
                       'job_id TEXT, operation TEXT, created_at TEXT, payload TEXT NOT NULL)')
            db.execute('CREATE INDEX IF NOT EXISTS diagnostic_events_job ON diagnostic_events(operation,job_id,id)')
            db.execute('CREATE INDEX IF NOT EXISTS diagnostic_events_trace ON diagnostic_events(trace_id,id)')
            db.execute('CREATE TABLE IF NOT EXISTS diagnostic_alerts ('
                       'id INTEGER PRIMARY KEY AUTOINCREMENT, dedupe_key TEXT, trace_id TEXT, '
                       'job_id TEXT, operation TEXT, scope TEXT, resolved_at TEXT, payload TEXT NOT NULL)')
            db.execute('CREATE UNIQUE INDEX IF NOT EXISTS diagnostic_alerts_open '
                       'ON diagnostic_alerts(dedupe_key) WHERE resolved_at IS NULL')

    def _upsert(self, db, table, item):
        old = db.execute(f'SELECT payload FROM diagnostic_{table} WHERE id=?', (item['id'],)).fetchone()
        payload = {**(json.loads(old['payload']) if old else {}), **item}
        payload.pop('kind', None)
        payload['trace_id'] = payload.get('trace_id') or payload['id']
        columns = ('id', 'trace_id', 'scope', 'job_id', 'operation', 'status', 'started_at')
        values = [payload.get(key) for key in columns] + [json.dumps(payload, ensure_ascii=False, allow_nan=False)]
        db.execute(f'INSERT INTO diagnostic_{table} ({",".join(columns)},payload) VALUES (?,?,?,?,?,?,?,?) '
                   'ON CONFLICT(id) DO UPDATE SET trace_id=excluded.trace_id,scope=excluded.scope,'
                   'job_id=excluded.job_id,operation=excluded.operation,status=excluded.status,'
                   'started_at=excluded.started_at,payload=excluded.payload', values)

    def write(self, item):
        with self.connection(write=True) as db:
            kind = item['kind']
            if kind=='gap':
                old=db.execute('SELECT payload FROM diagnostic_traces WHERE id=?',(item['id'],)).fetchone()
                payload=json.loads(old['payload']) if old else {'id':item['id'],'scope':item.get('scope','system'),
                    'status':'unknown','started_at':item.get('created_at',utc_now())}
                payload['diagnostics_incomplete']=True
                self._upsert(db,'traces',payload)
                return
            if kind in ('trace', 'span'):
                self._upsert(db, kind + 's', item)
                if item.get('status') not in (None, 'running', 'pending'):
                    self._resolve(db, span_id=item['id'], rules=['SLOW_OPERATION'])
                return
            if kind == 'resolve':
                self._resolve(db, trace_id=item.get('trace_id'), job_id=item.get('job_id'),
                              operation=item.get('operation'), rules=item.get('rules'), stage=item.get('resolve_stage'))
                return
            if kind == 'alert':
                self._alert(db, item)
                return
            if kind == 'event':
                payload = {k: v for k, v in item.items() if k != 'kind'}
                db.execute('INSERT OR IGNORE INTO diagnostic_events '
                           '(event_key,trace_id,job_id,operation,created_at,payload) VALUES (?,?,?,?,?,?)',
                           (item['event_key'], item.get('trace_id'), item.get('job_id'), item.get('operation'),
                            item['created_at'], json.dumps(payload, ensure_ascii=False, allow_nan=False)))

    def _alert(self, db, item):
        if item.get('rule')=='SLOW_OPERATION':
            # A checker can race the operation's completion. Never reopen a slow
            # incident from a stale in-memory snapshot after its end was stored.
            found = db.execute('SELECT status FROM diagnostic_spans WHERE id=? UNION ALL '
                               'SELECT status FROM diagnostic_traces WHERE id=?',
                               (item.get('span_id'),item.get('span_id'))).fetchall()
            if not found or all(row['status'] not in ('running','pending') for row in found):
                return
        key = item['dedupe_key']
        old = db.execute('SELECT id,payload FROM diagnostic_alerts WHERE dedupe_key=? AND resolved_at IS NULL', (key,)).fetchone()
        if old:
            payload = json.loads(old['payload'])
            if item.get('once'):
                return
            payload['occurrences'] += 1
            payload['last_seen_at'] = utc_now()
            db.execute('UPDATE diagnostic_alerts SET payload=? WHERE id=?', (json.dumps(payload), old['id']))
            return
        payload = {k: v for k, v in item.items() if k not in ('kind', 'once')}
        payload.update(first_seen_at=utc_now(), last_seen_at=utc_now(), occurrences=1,
                       acknowledged_at=None, acknowledged_by=None, resolved_at=None)
        db.execute('INSERT INTO diagnostic_alerts (dedupe_key,trace_id,job_id,operation,scope,payload) VALUES (?,?,?,?,?,?)',
                   (key, item.get('trace_id'), item.get('job_id'), item.get('operation'), item.get('scope'), json.dumps(payload)))

    def _resolve(self, db, *, trace_id=None, span_id=None, job_id=None, operation=None, rules=None, stage=None):
        if not any((trace_id, span_id, job_id)):
            return
        for row in db.execute('SELECT id,payload FROM diagnostic_alerts WHERE resolved_at IS NULL').fetchall():
            payload = json.loads(row['payload'])
            if rules and payload.get('rule') not in rules:
                continue
            if stage is not None and payload.get('stage') != stage:
                continue
            matches = (payload.get('span_id') == span_id if span_id else
                       (payload.get('job_id') == job_id and payload.get('operation') == operation) if job_id else
                       payload.get('trace_id') == trace_id)
            if job_id and payload.get('rule')=='BLOCKED_BY_RECOVERY' and payload.get('blocked_by_job_id')==job_id:
                matches=True
            if matches:
                payload['resolved_at'] = utc_now()
                db.execute('UPDATE diagnostic_alerts SET resolved_at=?,payload=? WHERE id=?',
                           (payload['resolved_at'], json.dumps(payload), row['id']))

    def acknowledge(self, alert_id, actor):
        with self.connection(write=True) as db:
            row = db.execute('SELECT payload FROM diagnostic_alerts WHERE id=?', (alert_id,)).fetchone()
            if row is None:
                return None
            payload = json.loads(row['payload'])
            if not payload.get('acknowledged_at'):
                payload.update(acknowledged_at=utc_now(), acknowledged_by=str(actor)[:128])
                db.execute('UPDATE diagnostic_alerts SET payload=? WHERE id=?', (json.dumps(payload), alert_id))
            return {**self._public(payload), 'id': alert_id}

    @staticmethod
    def _public(payload):
        return {key: value for key, value in payload.items() if key not in ('user_id', 'session_id', 'run_id')}

    def traces(self, *, scope='chat', before_id=None, limit=50, status=None, since=None):
        conditions, params = ['scope=?'], [scope]
        if before_id is not None:
            conditions.append('rowid<?'); params.append(before_id)
        if status:
            conditions.append('status=?'); params.append(status)
        if since:
            conditions.append('started_at>=?'); params.append(since)
        with self.connection() as db:
            rows = db.execute('SELECT rowid AS cursor,payload FROM diagnostic_traces WHERE ' + ' AND '.join(conditions) +
                              ' ORDER BY rowid DESC LIMIT ?', (*params, limit + 1)).fetchall()
        return {'items': [self._public(json.loads(row['payload'])) for row in rows[:limit]],
                'next_cursor': rows[limit-1]['cursor'] if len(rows) > limit else None}

    def trace_detail(self, trace_id):
        with self.connection() as db:
            row = db.execute('SELECT payload FROM diagnostic_traces WHERE id=?', (trace_id,)).fetchone()
            if not row:
                return None
            spans = db.execute('SELECT payload FROM diagnostic_spans WHERE trace_id=? ORDER BY rowid LIMIT 1000', (trace_id,)).fetchall()
            alerts = db.execute('SELECT id,payload FROM diagnostic_alerts WHERE trace_id=? ORDER BY id DESC LIMIT 100', (trace_id,)).fetchall()
        return {'trace': self._public(json.loads(row['payload'])),
                'spans': [self._public(json.loads(row['payload'])) for row in spans],
                'events': self.events(trace_id=trace_id, limit=100)['items'],
                'alerts': [{**self._public(json.loads(row['payload'])), 'id':row['id']} for row in alerts]}

    def document_attempts(self, operation, job_id):
        with self.connection() as db:
            rows = db.execute('SELECT id FROM diagnostic_traces WHERE scope=? AND operation=? AND job_id=? '
                              'ORDER BY rowid DESC LIMIT 100', ('document', operation, job_id)).fetchall()
        result = []
        for row in rows:
            detail = self.trace_detail(row['id'])
            # Recovery runs have their own trace, but their job spans still belong
            # in that document's history after the original process exits.
            with self.connection() as db:
                spans = db.execute('SELECT payload FROM diagnostic_spans WHERE operation=? AND job_id=? '
                                   "AND json_extract(payload,'$.attempt')=? ORDER BY rowid LIMIT 1000",
                                   (operation,job_id,detail['trace'].get('attempt',1))).fetchall()
            result.append({**detail['trace'], 'spans':[self._public(json.loads(item['payload'])) for item in spans]})
        return result

    def events(self, *, trace_id=None, operation=None, job_id=None, attempt=None, after_id=0, limit=50):
        conditions, params = ['id>?'], [after_id]
        for column, value in [('trace_id', trace_id), ('operation', operation), ('job_id', job_id)]:
            if value is not None:
                conditions.append(column+'=?'); params.append(value)
        if attempt is not None:
            conditions.append("json_extract(payload,'$.attempt')=?"); params.append(attempt)
        with self.connection() as db:
            rows = db.execute('SELECT id,payload FROM diagnostic_events WHERE ' + ' AND '.join(conditions) +
                              ' ORDER BY id LIMIT ?', (*params, limit+1)).fetchall()
        return {'items':[{**self._public(json.loads(row['payload'])), 'id':row['id']} for row in rows[:limit]],
                'next_cursor': rows[limit-1]['id'] if len(rows) > limit else None}

    def alerts(self, *, state='open', before_id=None, limit=50, scope=None):
        conditions, params = ['1=1'], []
        if state != 'all':
            conditions.append('resolved_at IS ' + ('NULL' if state == 'open' else 'NOT NULL'))
        if before_id is not None:
            conditions.append('id<?'); params.append(before_id)
        if scope:
            conditions.append('scope=?'); params.append(scope)
        with self.connection() as db:
            rows = db.execute('SELECT id,payload FROM diagnostic_alerts WHERE '+' AND '.join(conditions)+
                              ' ORDER BY id DESC LIMIT ?', (*params, limit+1)).fetchall()
        return {'items':[{**self._public(json.loads(row['payload'])), 'id':row['id']} for row in rows[:limit]],
                'next_cursor':rows[limit-1]['id'] if len(rows)>limit else None}

    def summary(self, *, scope='document', hours=24):
        since = (datetime.now(timezone.utc)-timedelta(hours=hours)).isoformat()
        with self.connection() as db:
            counts = {row['status']: row['n'] for row in db.execute(
                'SELECT status,COUNT(*) AS n FROM diagnostic_traces WHERE scope=? AND started_at>=? GROUP BY status', (scope, since))}
            rows = db.execute('SELECT payload FROM diagnostic_spans WHERE scope=? AND started_at>=?', (scope,since)).fetchall()
        groups = {}
        for row in rows:
            item = json.loads(row['payload'])
            key = tuple(item.get(k) for k in ('stage','purpose','model','provider','status','cold_start')) + (item.get('mode') or item.get('retrieval_mode'),)
            bucket = groups.setdefault(key, {'values':[], 'unknown_count':0})
            duration = item.get('duration_ms')
            if duration is None or item.get('status') not in ('completed','failed','degraded','completed_after_cancel'):
                bucket['unknown_count'] += 1
            else:
                bucket['values'].append(duration)
        result = []
        for key, bucket in groups.items():
            values = sorted(bucket['values']); count=len(values)
            result.append({**dict(zip(('stage','purpose','model','provider','status','cold_start','mode'),key)),
                           'count':count, 'unknown_count':bucket['unknown_count'],
                           'avg_ms':sum(values)/count if count else None,
                           'p50_ms':values[max(0,math.ceil(count*.5)-1)] if count else None,
                           'p95_ms':values[max(0,math.ceil(count*.95)-1)] if count else None,
                           'insufficient_samples':count<20})
        return {'counts':counts, 'groups':result}

    def mark_interrupted(self, run_id):
        with self.connection(write=True) as db:
            for table in ('traces','spans'):
                rows = db.execute(f"SELECT payload FROM diagnostic_{table} WHERE status IN ('running','pending')").fetchall()
                for row in rows:
                    item = json.loads(row['payload'])
                    if item.get('run_id') != run_id:
                        item.update(status='interrupted', ended_at=utc_now(), duration_ms=None, diagnostics_incomplete=True)
                        self._upsert(db,table,item)
                        self._resolve(db,span_id=item['id'],rules=['SLOW_OPERATION'])

    def prune(self, days=30, max_records=100000):
        cutoff=(datetime.now(timezone.utc)-timedelta(days=days)).isoformat()
        # Bounded transactions; active traces and unresolved evidence are protected.
        with self.connection(write=True) as db:
            count=sum(db.execute(f'SELECT COUNT(*) FROM diagnostic_{table}').fetchone()[0]
                      for table in ('traces','spans','events','alerts'))
            rows=db.execute("SELECT id FROM diagnostic_traces t WHERE status NOT IN ('pending','running') "
                            'AND (started_at<? OR ? > ?) AND NOT EXISTS '
                            '(SELECT 1 FROM diagnostic_alerts a WHERE a.resolved_at IS NULL AND '
                            '(a.trace_id=t.id OR (a.job_id=t.job_id AND a.operation=t.operation))) '
                            "AND NOT EXISTS (SELECT 1 FROM diagnostic_spans s WHERE (s.trace_id=t.id OR "
                            "(s.job_id=t.job_id AND s.operation=t.operation)) AND s.status IN ('running','pending')) "
                            'ORDER BY started_at LIMIT 100', (cutoff,count,max_records)).fetchall()
            for row in rows:
                for table in ('events','spans','alerts'):
                    db.execute(f'DELETE FROM diagnostic_{table} WHERE trace_id=?', (row['id'],))
                db.execute('DELETE FROM diagnostic_traces WHERE id=?', (row['id'],))
            remaining=sum(db.execute(f'SELECT COUNT(*) FROM diagnostic_{table}').fetchone()[0]
                          for table in ('traces','spans','events','alerts'))
            return {'removed':len(rows), 'over_capacity':remaining>max_records}
