import json
import subprocess
import threading
import time
from html.parser import HTMLParser
from pathlib import Path

from backend import upload_jobs
from backend.document_loader import DocumentLoader


ROOT = Path(__file__).parent


class UploadMarkupParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.attrs = []

    def handle_starttag(self, tag, attrs):
        self.attrs.append((tag, dict(attrs)))


def test_markdown_document_is_supported_and_split_into_searchable_chunks(tmp_path):
    markdown = tmp_path / "guide.md"
    markdown.write_text(
        "# StarMew 使用指南\n\n这是 Markdown 正文。\n\n## 上传\n\n支持文件夹批量上传。",
        encoding="utf-8",
    )

    assert DocumentLoader.is_supported("guide.md") is True
    assert DocumentLoader.is_supported("GUIDE.MARKDOWN") is True

    chunks = DocumentLoader(chunk_size=500, chunk_overlap=50).load_document(
        str(markdown), markdown.name
    )

    assert chunks
    assert {chunk["file_type"] for chunk in chunks} == {"Markdown"}
    assert {chunk["chunk_level"] for chunk in chunks} == {1, 2, 3}
    assert all(chunk["filename"] == "guide.md" for chunk in chunks)
    assert any("支持文件夹批量上传" in chunk["text"] for chunk in chunks)


def test_markdown_loader_accepts_common_windows_chinese_encoding(tmp_path):
    markdown = tmp_path / "windows-notes.md"
    markdown.write_bytes("# 项目说明\n\n这是 GB18030 编码的文档。".encode("gb18030"))

    chunks = DocumentLoader().load_document(str(markdown), markdown.name)

    assert any("GB18030 编码" in chunk["text"] for chunk in chunks)


def test_upload_filename_validation_normalizes_folder_paths_and_rejects_unsafe_basename():
    assert DocumentLoader.validate_filename("项目说明.md") == "项目说明.md"
    assert DocumentLoader.validate_filename("资料/readme.md") == "readme.md"
    assert DocumentLoader.validate_filename("资料\\readme.md") == "readme.md"

    for unsafe in (
        '资料/bad"name.md', "资料/bad\nname.md",
        "CON.md", "nul.MD", "LPT1.markdown", "trailing .md.",
    ):
        try:
            DocumentLoader.validate_filename(unsafe)
        except ValueError:
            continue
        raise AssertionError(f"unsafe filename accepted: {unsafe!r}")


def test_upload_processor_gate_serializes_concurrent_jobs():
    active = 0
    max_active = 0
    state_lock = threading.Lock()
    start = threading.Barrier(3)

    def operation():
        nonlocal active, max_active
        with state_lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.03)
        with state_lock:
            active -= 1

    def worker():
        start.wait()
        upload_jobs.run_serialized_index_mutation(operation)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    start.wait()
    for thread in threads:
        thread.join()

    assert max_active == 1


def test_staging_paths_are_unique_for_same_named_uploads(tmp_path):
    first = upload_jobs.create_upload_staging_path(tmp_path, "job-one", "guide.md")
    second = upload_jobs.create_upload_staging_path(tmp_path, "job-two", "guide.md")

    assert first != second
    assert first.name == second.name == "guide.md"
    assert first.parent.name == "job-one"
    assert second.parent.name == "job-two"
    assert first.parent.parent == second.parent.parent == tmp_path / ".staging"


def test_folder_selection_filters_supported_files_and_batch_continues_after_failure():
    harness = r"""
const vm = require('node:vm');
const fs = require('node:fs');
let options;
const mediaQuery = { matches: false, addEventListener() {}, removeEventListener() {} };
vm.runInNewContext(fs.readFileSync(0, 'utf8'), {
  Vue: { createApp(value) { options = value; return { mount() {} }; } },
  window: {
    matchMedia() { return mediaQuery; },
    addEventListener() {}, removeEventListener() {}, setTimeout,
  },
  document: { documentElement: { dataset: {} }, createElement() { return {}; } },
  localStorage: { getItem() { return null; }, setItem() {}, removeItem() {} },
  marked: { setOptions() {}, parse(value) { return value; } },
  hljs: { getLanguage() { return false; }, highlight(value) { return { value }; } },
  fetch: async () => ({ ok: true, json: async () => ({}) }),
  XMLHttpRequest: function () {}, FormData: function () {},
  setInterval, clearInterval, setTimeout, clearTimeout,
});

const state = options.data();
for (const [name, method] of Object.entries(options.methods)) state[name] = method.bind(state);
const toasts = [];
state.showToast = (message, type) => toasts.push({ message, type });
state.$refs = { fileInput: { value: 'files' }, folderInput: { value: 'folder' } };

state.handleFileSelect({ target: { files: [
  { name: 'a.pdf', size: 1, type: 'application/pdf' },
  { name: 'notes.MD', size: 2, type: '' },
  { name: 'ignore.png', size: 3, type: 'image/png' },
] } });
const selected = state.selectedFiles.map(file => file.name);

state.uploadBatchCurrent = 2;
state.uploadBatchTotal = 5;
state.syncUploadJob({
  job_id: 'running-job', status: 'running', message: '正在解析文档', steps: state.createUploadSteps(),
});
const prefixedProgress = state.uploadProgress;

let pollAttempts = 0;
let terminalPollAttempts = 0;
let unknownPollAttempts = 0;
state.authFetch = async () => {
  pollAttempts += 1;
  if (pollAttempts < 3) throw new Error('temporary offline');
  return { ok: true, json: async () => ({
    job_id: 'retry-job', status: 'completed', message: 'done', steps: state.createUploadSteps(),
  }) };
};
state.waitBeforeUploadPoll = async () => {};
const waitForUploadJobWithRetry = state.waitForUploadJob.bind(state);

const started = [];
state.uploadFileWithProgress = async file => {
  started.push(file.name);
  if (file.name === 'a.pdf') throw new Error('network');
  return { job_id: 'job-' + file.name, message: 'queued' };
};
state.waitForUploadJob = async jobId => ({
  job_id: jobId, status: 'completed', message: 'done', steps: state.createUploadSteps(),
});
state.loadDocuments = async () => {};

const pausedState = options.data();
for (const [name, method] of Object.entries(options.methods)) pausedState[name] = method.bind(pausedState);
pausedState.selectedFiles = [
  { name: 'current.md', size: 1 },
  { name: 'remaining-a.md', size: 1 },
  { name: 'remaining-b.pdf', size: 1 },
];
pausedState.$refs = { fileInput: { value: 'files' }, folderInput: { value: 'folder' } };
pausedState.showToast = () => {};
pausedState.uploadFileWithProgress = async file => ({ job_id: 'job-' + file.name });
pausedState.waitForUploadJob = async jobId => {
  const error = new Error('status unknown');
  error.isUploadStatusUnknown = true;
  error.jobId = jobId;
  throw error;
};
pausedState.loadDocuments = async () => {};

(async () => {
  await waitForUploadJobWithRetry('retry-job');
  state.authFetch = async () => {
    terminalPollAttempts += 1;
    return { ok: true, json: async () => ({
      job_id: 'failed-job', status: 'failed', message: 'parse failed', steps: state.createUploadSteps(),
    }) };
  };
  let terminalError = '';
  try { await waitForUploadJobWithRetry('failed-job'); } catch (error) { terminalError = error.message; }
  state.authFetch = async () => {
    unknownPollAttempts += 1;
    throw new Error('offline');
  };
  let statusUnknown = false;
  try { await waitForUploadJobWithRetry('unknown-job'); } catch (error) {
    statusUnknown = error.isUploadStatusUnknown === true;
  }
  await state.uploadDocument();
  await pausedState.uploadDocument();
  process.stdout.write(JSON.stringify({
    selected,
    prefixedProgress,
    pollAttempts,
    terminalPollAttempts,
    terminalError,
    unknownPollAttempts,
    statusUnknown,
    started,
    remaining: state.selectedFiles.length,
    isUploading: state.isUploading,
    progress: state.uploadProgress,
    failures: state.uploadFailures,
    pausedRemaining: pausedState.selectedFiles.map(file => file.name),
    pausedProgress: pausedState.uploadProgress,
    toasts,
  }));
})().catch(error => { console.error(error); process.exitCode = 1; });
"""
    result = subprocess.run(
        ["node", "-e", harness],
        input=(ROOT / "frontend" / "script.js").read_text(encoding="utf-8"),
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=True,
    )
    runtime = json.loads(result.stdout)

    assert runtime["selected"] == ["a.pdf", "notes.MD"]
    assert runtime["prefixedProgress"] == "[2/5] 正在解析文档"
    assert runtime["pollAttempts"] == 3
    assert runtime["terminalPollAttempts"] == 1
    assert runtime["terminalError"] == "parse failed"
    assert runtime["unknownPollAttempts"] == 3
    assert runtime["statusUnknown"] is True
    assert runtime["started"] == ["a.pdf", "notes.MD"]
    assert runtime["remaining"] == 0
    assert runtime["isUploading"] is False
    assert "成功 1 个，失败 1 个" in runtime["progress"]
    assert runtime["failures"] == [{"filename": "a.pdf", "message": "network"}]
    assert runtime["pausedRemaining"] == ["remaining-a.md", "remaining-b.pdf"]
    assert "剩余 2 个文件" in runtime["pausedProgress"]
    assert runtime["toasts"][-1]["type"] == "warning"


def test_upload_controls_expose_multiple_file_and_recursive_folder_pickers():
    parser = UploadMarkupParser()
    parser.feed((ROOT / "frontend" / "index.html").read_text(encoding="utf-8"))
    inputs = {attrs.get("ref"): attrs for tag, attrs in parser.attrs if tag == "input"}

    assert "multiple" in inputs["fileInput"]
    assert ".md" in inputs["fileInput"]["accept"]
    assert "multiple" in inputs["folderInput"]
    assert "webkitdirectory" in inputs["folderInput"]
    assert "directory" in inputs["folderInput"]

    symbols = {attrs.get("id") for tag, attrs in parser.attrs if tag == "symbol"}
    assert "icon-folder" in symbols
