# Ragas 评估结果与缺失值约定

更新于 2026-10-05。适用于 `backend/evaluation.py`、单条/批量/历史评估 API 及其 JSON、CSV 导出。

## 五项指标

| 字段 | 含义 | 必需输入 |
|---|---|---|
| `faithfulness` | 回答是否有上下文支持 | 问题、回答、上下文 |
| `answer_relevancy` | 回答与问题的相关性 | 问题、回答 |
| `context_precision` | 上下文相关性及排序质量 | 问题、参考答案、上下文 |
| `context_recall` | 上下文对参考答案的覆盖 | 问题、参考答案、上下文 |
| `factual_correctness` | 回答相对参考答案的事实 F1 | 回答、参考答案 |

这里的 ContextPrecision 使用带参考答案的 Ragas 指标。没有参考答案时，不会改用无参考版本。`factual_correctness(mode=f1)` 会归一化为 `factual_correctness`，precision/recall 模式不会被当作 F1。

`answer_correctness` 是兼容旧客户端的独立、已弃用字段，不是 `factual_correctness` 的别名。新版客户端应读取 `metrics.factual_correctness`。

## 有效值和状态

有效分数必须是 `[0, 1]` 内的有限数值；布尔值、字符串、NaN、Infinity、越界值均无效。真实的 `0` 和 `0.5` 都是有效观测。

- 评分失败、超时、缺少返回值或无效分数：该项为 `null`，`metric_status` 为 `failed`，`metric_errors` 记录原因。
- 必需输入缺失：该项为 `null`，状态为 `not_applicable`，不调用该指标。
- 有效观测：状态为 `success`。
- SDK 不可用时的备用 LLM 评分同样遵循此约定，不再填入默认 `0.5`。

每条结果的 `evaluation_status` 含义：

| 状态 | 含义 |
|---|---|
| `success` | 本次适用指标全部成功；不代表五项都适用 |
| `partial` | 部分适用指标成功，部分失败 |
| `failed` | 没有适用指标成功，或上游 Pipeline 失败 |
| `not_applicable` | 没有适用指标 |
| `unknown` | 旧记录没有足够状态信息 |

`ragas_native=true/false` 表示该记录选择了 SDK / 备用评分路径，不能单独证明评分成功；应同时检查状态。Pipeline 在评分前失败时，该字段为 `null`。汇总级别的 `ragas_native` 仍表示当前环境的 SDK 可用性。

## 统计与导出

- 单项均值只使用该项有效且成功的观测；没有观测时均值为 `null`。
- `valid_counts` 给出五项指标各自的有效样本数。
- `overall_score` 按本条有效指标的既有权重归一化计算；全部缺失时为 `null`。
- `avg_overall_score` 是有效单条综合分的均值，分母见 `valid_overall_count`。
- `total_evaluations` 包含失败记录；`evaluation_status_counts` 公开各状态数量。
- 部分指标缺失的综合分使用不同指标集合，比较实验时应同时检查逐项均值和有效样本数。
- JSON 使用 `null`；CSV 分数字段为空，并附带状态、错误和评分路径；终端摘要显示 `N/A`。

例如，三条样本的忠实度分别为 `0`、`1`、评分失败，均值为 `0.5`、有效数为 `2`、总数为 `3`。失败项不会作为 `0` 或 `0.5` 加入均值。

批量评估直接持有各自的结果对象，不再通过共享 `history[-1]` 查找本条结果；Pipeline 异常也会进入历史和总样本数。

## 历史与验证边界

这次修复不回写已保存的测评 JSON，也没有重跑真实模型评测。代码回归测试覆盖评分异常、部分结果、输入适用性、API 序列化、均值分母、批量记录归属、导出和续跑脚本。

旧记录缺少状态且没有记录错误时，可沿用其有限有效分数进行汇总；若已有指标错误或失败状态，则排除该观测。不能仅凭旧分数等于 `0.5` 就断言它是伪造值。事实正确性续跑不会恢复其他指标的失败分数。

后续已使用当前阈值 6 重新生成回答和上下文，并完成五项真实评分，结果及有效分母见 [当前阈值 Ragas 重测](RAGAS_THRESHOLD6_RETEST.md)。此前的检索 Precision/Recall 复测与这些 Ragas 指标口径不同。实验比较流程见 [配对测评手册](RAG_COMPARISON.md)。
