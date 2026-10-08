# RAG 与无 RAG 配对测评手册

## 用途与结论边界

本流程在相同问题、生成模型、生成参数和评分规则下，比较 `no_rag` 与 `rag` 两组。它评估的是“当前 RAG 检索流程＋受控生成器”，不等于完整线上 Agent 的总体收益。

工具不会自动上传、删除或重建知识库。测评期间必须冻结 Milvus、PostgreSQL、BM25 状态及本地语料。

## 环境与身份冻结

在 `SuperMew` 项目目录执行。默认使用本机 Ollama，真实生成需要：

- Ollama 已启动且已安装 `.env` 中 `OLLAMA_MODEL` 指定的模型；
- 生成模型默认为 `OLLAMA_MODEL=qwen3.5:4b`；未配置 `EVALUATION_PROVIDER` 时测评裁判跟随应用裁判，优先使用 `OLLAMA_JUDGE_MODEL`；
- 仅将测评切到阿里云：设置 `EVALUATION_PROVIDER=aliyun`、`GRADE_MODEL=qwen3.7-flash`，复用 `.env` 中的 `BASE_URL` 和 `ARK_API_KEY`（历史变量名）。应用的 `LLM_PROVIDER=ollama` 和 `OLLAMA_*` 保持不变，RAG 内部相关性评分仍走本地。该设置同时适用于比较脚本和 `/evaluation/run`；测评问题、答案、参考答案及需要的上下文会发送至配置的云端接口；
- `--judge-model` 只覆盖模型名，不切换接口。裁判提供方和接口记录在运行指纹中，换裁判后不能直接续跑旧目录；使用新的评分版本/目录，保留旧结果；
- 仅在 `LLM_PROVIDER=ark` 时读取旧 `MODEL`、`GRADE_MODEL`、`BASE_URL`、`ARK_API_KEY` 配置；
- 可连接的 Milvus、PostgreSQL 和本地 BGE-m3；
- 已索引的语料，不使用旧 `ensure_document_indexed` 自动上传流程。

`manifest.json` 保存数据集、源码、`uv.lock`、提示词、BM25 状态和文档语料的哈希，以及 Git/dirty 状态、Ragas 版本、实际模型与端点主机标识。密钥不会写入报告。指纹变化时 `resume` 会拒绝混合续跑。

## 数据准备

先转换旧数据，转换结果默认仍为 exploratory：

```powershell
uv run python scripts/run_rag_comparison.py validate --dataset data/eval_samples_dishes.json --output data/rag_comparison_dataset_dishes_exploratory_v1.json
```

菜谱集先自动关联源文档、记录证据范围，并按来源组拆分 dev/test。自动抽取不是人工审核，产物保持 `reference_reviewed=false`，不能直接用于正式测评：

```powershell
uv run python scripts/audit_dishes_comparison_dataset.py --input data/eval_samples_dishes.json --source-root D:\Desktop\supermew改版 --output data/rag_comparison_dataset_dishes_v2_draft.json --dev-target 20 --seed 42 --audit-date 2026-09-25
uv run python scripts/export_dishes_review_sheet.py --dataset data/rag_comparison_dataset_dishes_v2_draft.json --output data/rag_comparison_dataset_dishes_v2_review.csv
```

逐题填写复核表中的 `review_decision`（`approve` 或 `revise`）、`reviewer`、`reviewed_at`；修订题目或答案时填写 `corrected_question` / `corrected_ground_truth`。尤其检查全部负例、所有允许多种正确细节的题，以及参考答案仅为链接的题。全部 120 题人工确认后，可冻结这批**已曝光题目的已审版本**：

```powershell
uv run python scripts/finalize_dishes_review.py --draft data/rag_comparison_dataset_dishes_v2_draft.json --review-sheet data/rag_comparison_dataset_dishes_v2_review.csv --output data/rag_comparison_dataset_dishes_v3.json --dataset-version dishes-human-reviewed-v3
uv run python scripts/run_rag_comparison.py validate --dataset data/rag_comparison_dataset_dishes_v3.json --output data/rag_comparison_dataset_dishes_v3_validated.json --formal
```

`v1` 自动标记已审核，但未进行逐题人工确认；其已有运行只可作诊断。调整问题、答案、证据或裁判规则后，必须创建新数据/评分版本和新运行目录，不能混入旧运行。

注意：旧版 100 道 test 题已执行双组生成，其中部分答案参与错误排查；`v3` 即使通过 `--formal` 的结构校验，也**不是未见过的正式留出测试集**。`--formal` 只验证字段与人工审核标记，无法验证题目是否参与过调试。`v3` 只能用于开发/探索性测评。若要发表计划中的正式增益结论，须按来源组另建、审核、冻结未参与调试的新题集，使用新运行目录，并在报告中记录题目来源与未曝光证明。下文正式测试命令仅适用于该新留出集，不能把 `v3` 文件代入。

## 开发集预演

先检查命令参数：

```powershell
uv run python scripts/run_rag_comparison.py --help
uv run python scripts/run_rag_comparison.py run --help
uv run python scripts/run_rag_comparison.py resume --help
uv run python scripts/run_rag_comparison.py report --help
```

先用 2 道 dev 题进行连通性检查：

```powershell
uv run python scripts/run_rag_comparison.py run --dataset data/rag_comparison_dataset_dishes_v3.json --split dev --limit 2 --repeat 1 --seed 42 --metrics factual_correctness,abstention --score-concurrency 2 --formal --run-dir data/rag_comparison/dishes_connectivity_v3
```

连通后再跑约 20 道开发题：

```powershell
uv run python scripts/run_rag_comparison.py run --dataset data/rag_comparison_dataset_dishes_v3.json --split dev --limit 20 --repeat 1 --seed 42 --metrics factual_correctness,abstention --diagnostics faithfulness,context_precision,context_recall --score-concurrency 2 --formal --run-dir data/rag_comparison/dishes_dev_v3
```

生成完成后才会启动评分。无 RAG 记录的 `retrieved_docs` 和 `used_contexts` 必须为空；RAG 最终 `prompt` 中的资料必须与 `used_contexts` 一致。

## 正式测试集（仅限新建的未曝光留出集）

先只生成回答，再单独评分：

```powershell
uv run python scripts/run_rag_comparison.py run --dataset data/rag_comparison_dataset_dishes_holdout_v4.json --split test --formal --repeat 1 --seed 42 --metrics factual_correctness,abstention --phase generate --run-dir data/rag_comparison/dishes_test_holdout_v4
uv run python scripts/run_rag_comparison.py resume --run-dir data/rag_comparison/dishes_test_holdout_v4 --phase score
```

如在运行前冻结为 3 次重复，将 `--repeat 3` 写在创建新 run 的命令中。不允许看到结果后只给选定题补跑。

## 断点、失败和退出码

```powershell
uv run python scripts/run_rag_comparison.py resume --run-dir data/rag_comparison/dishes_test_holdout_v4 --retry-failed
uv run python scripts/run_rag_comparison.py report --run-dir data/rag_comparison/dishes_test_holdout_v4
```

- `0`：本次选中阶段全部完成。
- `1`：存在生成/评分失败或未完成项。
- `2`：参数、数据或指纹错误。

已成功生成的答案不会因评分失败而重新生成。`--retry-failed` 保留旧 attempts，且不超过 manifest 冻结的总尝试上限。运行目录有排他锁，不允许两个进程同时续跑。

## 输出和解读

`data/rag_comparison/<run_id>/` 包含：

- `manifest.json`：冻结配置、身份和计划队列；
- `dataset.json`：本次样本快照；
- `checkpoint.json`：每个 `(sample_id, repeat_index, arm)` 的状态与尝试；
- `summary.json`、`pairs.csv`、`report.md`：机器统计、逐题记录和中文报告；
- `review.csv`、`review_key.json`：匿名人工复核表及解盲映射。

主指标是可回答题上的 factual correctness F1，不能称为“答对率”。报告使用同题有效配对、先题内平均、再按 `cluster_id` bootstrap。基线为零时相对增益为 null。有效来源组少于 10 或主指标覆盖率低于 95% 时，结论必须标为探索性。

不可回答题的正确拒答率、可回答题的过度拒答率、生成失败、评分失败、耗时和 RAG 诊断分开报告。耗时不含裁判。检索内 LLM usage 不完整时，不生成“RAG 总费用增幅”。

## 人工复核

开发集预演审查全部负例和主要退步题。正式测试集按题型分层匿名抽查至少 20%，另外审查全部负例和差值最大的 10 道退步题。如发现裁判系统性偏差，创建新评分版本并对两组全部重评，不得挑题修分。

## 离线回归

```powershell
uv run --with pytest python -m pytest tests/test_eval_comparison_contracts.py tests/test_eval_comparison_generation.py tests/test_eval_comparison_scoring.py tests/test_eval_comparison_storage.py tests/test_eval_comparison_reporting.py tests/test_eval_comparison_cli.py test_evaluation_metric_normalization.py -q
```

这些测试使用合成数据和 fake 依赖，不应连接真实模型、Milvus、PostgreSQL 或嵌入模型。
