"""Timing must describe real model activity without retaining private content."""
import asyncio
import importlib
import unittest
import ast
from pathlib import Path
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch


class Operation:
    def __init__(self, stage, attrs):
        self.stage, self.attrs, self.status = stage, attrs, 'completed'
        self.first = 0
        self.id = str(id(self))
    def mark_first_content(self):
        self.first += 1
    def annotate(self, **attrs):
        self.attrs.update(attrs)
    def set_result(self, status):
        self.status = status


class ModelTimingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        try:
            self.diag = importlib.import_module('backend.chat_diagnostics')
        except ModuleNotFoundError:
            self.fail('chat diagnostics implementation is missing')
        self.operations = []
        @contextmanager
        def span(stage, **attrs):
            operation = Operation(stage, attrs)
            self.operations.append(operation)
            try:
                yield operation
            except BaseException as exc:
                operation.status = 'cancelled' if isinstance(exc, (asyncio.CancelledError, GeneratorExit)) else 'failed'
                raise
        self.patcher = patch.object(self.diag.telemetry, 'span', span)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    async def test_empty_tool_and_usage_chunks_are_not_first_content(self):
        model = SimpleNamespace(model_name='actual-model')
        with self.diag.model_call(model, purpose='final_answer', streaming=True) as call:
            call.observe(SimpleNamespace(content='', usage_metadata=None))
            call.observe(SimpleNamespace(content='private', tool_call_chunks=[{}]))
            self.assertEqual(self.operations[-1].first, 0)
            call.observe(SimpleNamespace(content=[{'type':'reasoning', 'text':'secret'}]))
            self.assertEqual(self.operations[-1].first, 0)
            call.observe(SimpleNamespace(content='answer'))
            call.observe(SimpleNamespace(content=' more'))
            call.observe(SimpleNamespace(content='', usage_metadata={'input_tokens': 4, 'output_tokens': 2, 'total_tokens':6}))
        op = self.operations[-1]
        self.assertEqual(op.first, 1)
        self.assertEqual(op.attrs['model'], 'actual-model')
        self.assertEqual(op.attrs['output_tokens'], 2)
        self.assertNotIn('secret', str(op.attrs))
        self.assertNotIn('answer more', str(op.attrs))
        self.assertIsNotNone(op.attrs['first_answer_ms'])

    async def test_nonstream_and_hidden_reason_have_no_answer_timing(self):
        model = SimpleNamespace(model_name='actual-model')
        with self.diag.model_call(model, purpose='tool_decision', streaming=True) as call:
            call.observe(SimpleNamespace(content='hidden'))
        with self.diag.model_call(model, purpose='grading') as call:
            call.observe(SimpleNamespace(content='yes'))
        self.assertEqual(self.operations[0].first, 1)
        self.assertIsNone(self.operations[0].attrs.get('first_answer_ms'))
        self.assertEqual(self.operations[1].first, 0)

    async def test_cancelled_stream_does_not_report_complete_generation(self):
        with self.assertRaises(asyncio.CancelledError):
            with self.diag.model_call(SimpleNamespace(model_name='m'), purpose='final_answer', streaming=True) as call:
                call.observe(SimpleNamespace(content='partial'))
                raise asyncio.CancelledError()
        self.assertEqual(self.operations[-1].status, 'cancelled')
        self.assertIsNone(self.operations[-1].attrs.get('generation_total_ms'))

    async def test_concurrent_requests_have_independent_ordinals(self):
        async def request():
            with self.diag.request_metrics():
                with self.diag.model_call(SimpleNamespace(), purpose='grading'):
                    await asyncio.sleep(0)
                with self.diag.model_call(SimpleNamespace(), purpose='grading'):
                    pass
        await asyncio.gather(request(), request())
        self.assertEqual([op.attrs['ordinal'] for op in self.operations], [1, 1, 2, 2])

    async def test_structured_grading_failure_records_both_actual_calls(self):
        from backend import telemetry
        from backend.agent_budget import current_run_state, remaining_timeout, model_timeout_kwargs
        from backend.rag_pipeline import GradeDocuments
        from backend.conversation_context import model_budget, assert_model_input, ContextBudgetExceeded, model_output_kwargs, structured_output_kwargs
        from backend.evidence_context import aggregate_evidence, select_evidence, render_evidence
        source = ast.parse((Path(__file__).parent / 'backend/rag_pipeline.py').read_text(encoding='utf-8'))
        node = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'grade_documents_node')
        class Grader:
            model_name = 'judge-model'
            calls = 0
            def with_structured_output(self, schema, **kwargs):
                return self
            def invoke(self, inputs, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise ValueError('private source body')
                return SimpleNamespace(content='yes')
        grader = Grader()
        namespace = dict(RAGState=dict, GradeDocuments=GradeDocuments, _get_grader_model=lambda: grader,
                         model_budget=model_budget, assert_model_input=assert_model_input,
                         ContextBudgetExceeded=ContextBudgetExceeded, model_output_kwargs=model_output_kwargs,
                         structured_output_kwargs=structured_output_kwargs,
                         aggregate_evidence=aggregate_evidence, select_evidence=select_evidence, render_evidence=render_evidence,
                         emit_rag_step=lambda *a:None, GRADE_PROMPT='{question} {context}',
                         model_call=self.diag.model_call, invoke_model=self.diag.invoke_model,
                         remaining_timeout=remaining_timeout, model_timeout_kwargs=model_timeout_kwargs,
                         current_run_state=current_run_state,
                         span=telemetry.span, timed=telemetry.timed)
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'rag_pipeline.py', 'exec'), namespace)
        from backend.run_control import check_cancelled, RunCancelled
        namespace.update(check_cancelled=check_cancelled, RunCancelled=RunCancelled)
        result = namespace['grade_documents_node']({'question':'private', 'context':'private', 'docs':[{'text': 'private'}]})
        self.assertEqual(result['route'], 'generate_answer')
        calls = [op for op in self.operations if op.attrs.get('purpose') == 'grading']
        self.assertEqual(len(calls), 2)
        self.assertEqual([op.status for op in calls], ['failed', 'completed'])
        self.assertEqual(calls[1].attrs['fallback_of'], calls[0].id)

    async def test_tool_error_is_recorded_without_arguments(self):
        def fail(inputs):
            raise RuntimeError('secret tool arguments')
        result = self.diag.invoke_tool(SimpleNamespace(invoke=fail), {'query':'private'}, purpose='web_search')
        self.assertEqual((result.status, result.code), ('error', 'UPSTREAM_ERROR'))
        self.assertNotIn('secret', result.content)
        self.assertEqual(self.operations[-1].stage, 'tool')
        self.assertEqual(self.operations[-1].status, 'failed')
        self.assertNotIn('private', str(self.operations[-1].attrs))


if __name__ == '__main__':
    unittest.main()
