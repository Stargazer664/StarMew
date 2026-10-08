"""Actionable rules: diagnostic evidence is broader than the alert inbox."""


def classify_alert(name, status, code, scope):
    if status not in ('failed','error','warning','degraded','blocked'):
        return None
    if code=='BLOCKED_BY_RECOVERY':
        return 'BLOCKED_BY_RECOVERY','critical','索引操作被未完成恢复阻塞'
    if name in ('recovery_failed','rollback_failed') or code in (
        'RECOVERY_FAILED','ROLLBACK_FAILED','JOURNAL_UNREADABLE','ROLLBACK_MANIFEST_FAILED','JOB_RECORD_MISSING'
    ) or (code or '').endswith('_RESTORE_FAILED'):
        return 'RECOVERY_FAILED','critical','恢复未完成，请检查失败阶段及依赖'
    if code=='STATUS_PERSIST_FAILED' or name=='status_failed':
        return 'STATUS_PERSIST_FAILED','error','业务数据与任务状态待核对，需补写任务状态'
    if 'cleanup' in name or 'CLEANUP' in (code or ''):
        return 'CLEANUP_FAILED','warning','数据操作已结束，但恢复文件或暂存文件清理失败'
    if name=='task_failed':
        return 'TASK_FAILED','warning','本次任务未完成，请查看原始错误与补偿结果'
    if code in ('RERANK_DEGRADED','HYBRID_FALLBACK'):
        return code,'warning','问答使用了降级路径，请查看失败调用'
    if code in ('MODEL_CALL_FAILED','MODEL_FAILED','RETRIEVAL_FAILED'):
        return code,'error','模型或检索调用失败，请查看对应操作'
    # Individual document store failures stay in the timeline. Their actionable
    # result is expressed by task_failed or recovery_failed after compensation.
    if scope=='chat' and code in ('REQUEST_FAILED','PROCESSING_FAILED'):
        return 'CHAT_FAILED','error','问答未能完整结束，请查看处理或保存阶段'
    return None
