# Solver-Informed Travel Agent V7.2

`solver_informed_travel_agent.py` 是单文件 agent/controller。运行时只允许
official tools 将事实写入 `EvidenceStore`，agent 不直接读取 case database。

当前主循环为：

```text
Typed ProblemIR
  → Independent Semantic Audit
  → Typed Constraint IR
  → Incremental SAT / UNSAT / UNKNOWN Propagation
  → Unified priority/dedup AcquisitionRequest Queue
  → Official Tool Adapters
  → Evidence + Fact Contracts
  → Fast Feasibility / Lack Check
  → Top-K transport/hotel macro plans
  → Global multi-day assignment + budget lower bound
  → Frontier-aware shared-fact acquisition
  → Lazy selected-edge route acquisition
  → Relax-and-Complete on selected-route stagnation
  → Deterministic earliest-time day scheduling
  → Conflict/no-good feedback
  → Stagnation-triggered Conflict Reflection:
      KEEP | RECHECK_SEMANTICS | EXPAND_RECALL
  → Targeted Retrieval + ReSolve
  → Structural verifier
  → ConvertedPlan JSON
  → text view
```

主要变化包括：时间/价格/时长 typed normalization、hard/soft/objective 分离、
显式工具 adapter、跨日实体唯一性与必选景点覆盖、候选级/边级 no-good、仅查询
frontier 中多个 provisional plans 的高价值共享事实，以及直接从结构化解生成
一次 benchmark JSON。Constraint engine 在路线检索前检查 transport/hotel/attraction/
restaurant coverage、预算 lower bound 和缺失事实。`UNSAT` 只有在可重放的
`InfeasibilityCertificate` 通过校验后才能升级为 `PROVEN_INFEASIBLE`。

V7 删除了 master 之后重复的 day-level candidate MILP。Global MILP 选定实体与顺序后，
day scheduler 仅执行可证明的 earliest-time propagation；若最早可达时间仍超过营业窗口
或 deadline，则任何更晚安排同样不可行。主文件也不再内置 ReAct fallback，避免主方法
与 baseline/repair variant 混杂。validator 只做 schema/type/enum normalization；自然语言
语义由 Parse → Semantic Audit/Reflection 负责，不再由中文 regex 暗中重写。

V7.2 在 deterministic recovery 无进展后使用最多两次受限 reflection。LLM 只查看
ProblemIR、solver conflict、候选/搜索摘要和 best-incumbent 摘要，只能返回
`KEEP | RECHECK_SEMANTICS | EXPAND_RECALL`。Recall 仍必须引用
grounded anchor 或 official attraction type，并由 controller 编译成 AcquisitionRequest。
LLM 无权声明 `PROVEN_INFEASIBLE`。若 reflection budget 用尽但已经存在可转换的
incumbent，则输出 best-effort artifact，并保持 `internal_verified=false` 以避免把交付
与 exact success 混淆。

查询状态显式区分 `SUCCESS_NONEMPTY / SUCCESS_EMPTY / ERROR / PARSE_ERROR`，并单独记录
`exhaustive`。只有成功且由 public tool contract 定义为 exhaustive 的证据，才能参与
证书。当前确定性证书仅包括：穷举域无满足者、exact entity absence、以及建立在完整
transport/hotel 域上的 budget lower bound。recommendation pool、模板容量、无 macro、
Gurobi infeasible 或 search exhaustion 均不是全局不可行证明。

V3.1 将正常路径从单次 parser 调用改为两阶段 neural-symbolic 编译：第一次
生成 IR，第二次只审计它是否与原请求语义等价。Audit 的候选修订不会被直接
信任：核心行程字段和交通模式不可修改，新增语义必须引用原文，明确
“必须/一定/must”的来源不得被删除。被 schema/type checker 拒绝的 Audit 最多允许
一次携带精确错误的局部重试。运行期语义修复和候选扩展已合并到一个 reflection
controller；后者只能引用已落地坐标 anchor 或 public schema 中的景点类型，并被编译
成普通 `AcquisitionRequest`。真实无候选不会自动放宽。

## 环境

在仓库根目录设置模型密钥：

```bash
export DEEPSEEK_API_KEY='your-key'
```

需要 benchmark 依赖、可用的 Gurobi 安装与 license。官方 tool schema、
`models_config.json` 和数据保持原目录结构。

## 运行 10 个 cases

```bash
BENCHMARK_DATA_DIR="$PWD/data" python -u travelplanning/GurobiAgent/solver_informed_travel_agent.py \
  --model deepseek-flash --language zh --workers 2 \
  --max-llm-calls 30 --max-tool-calls 400 --max-solver-rounds 64 \
  --ids 0 1 2 3 4 5 6 7 8 9 --evaluate \
  --output-dir travelplanning/results/solver_informed_v7_10_zh
```

## 全量 120 cases

省略 `--ids` 即读取中文 test data 的全部 cases：

```bash
BENCHMARK_DATA_DIR="$PWD/data" python -u travelplanning/GurobiAgent/solver_informed_travel_agent.py \
  --model deepseek-flash --language zh --workers 2 \
  --max-llm-calls 30 --max-tool-calls 400 --max-solver-rounds 64 \
  --evaluate \
  --output-dir travelplanning/results/solver_informed_v7_all_zh
```

`--evaluate` 会在推理后对本次选中的 IDs 运行官方 evaluator。ReAct baseline/repair
应由外层实验 harness 组合，不属于本 agent 的主路径。
默认启用独立 semantic audit 和条件式 reflection。消融实验可加入：

```bash
--no-semantic-audit
--no-reflection
```

`--no-semantic-recovery` 与 `--no-constraint-repair` 仅保留为
`--no-reflection` 的兼容 alias，不再代表独立模块。

Reflection 默认预算为 2，可通过 `--max-reflections 0..N` 调节。

## 离线测试

```bash
python -m py_compile travelplanning/GurobiAgent/solver_informed_travel_agent.py
python -m unittest discover -s travelplanning/GurobiAgent/tests -p 'test_*.py'
```

当前离线测试 33/33，通过且不调用模型 API，覆盖 typed IR、`24:00`、严格年份语义、soft/objective、
Typed Constraint IR、预算 lower-bound 推理、官方工具参数 adapter、餐厅坐标依赖、
跨日唯一性、validator 不做 query-regex 语义补全、确定性 earliest-time scheduler，
semantic revision 的 source grounding、核心字段与 mandatory-source 保护，以及
recall/reflection intent 到 official-tool request 的受限编译、重复 reflection 拒绝，以及
best-effort incumbent 不误标 exact success、query error 不构成完整域、certificate
fail-closed 校验，以及仅清理完全空的 parser constraint placeholder。

frontier 可通过环境变量调节：

```bash
export TRAVEL_FRONTIER_WIDTH=6
export TRAVEL_FACT_BATCH=32
```

前者控制每轮保留/探测的 provisional states，后者限制按共享次数和影响优先级
选出的事实请求数量。

## 输出与指标

输出目录包含：

```text
reports/
converted_plans/
trajectories/
summary.json
evaluation/evaluation_summary.json
```

`summary.json` 区分 `produced_plan`、`structured_ready`、`internal_verified`；兼容字段
`success` 等于 `internal_verified`，不再表示“任意非空文本”。官方成绩以
`evaluation/evaluation_summary.json` 为准。trajectory 还记录 `solver_rounds`、
`master_assignments`、`master_day_cuts`、`route_queries`、LLM/token/tool 指标和
结构化冲突。新增 `parser_attempts`、`parser_validation_errors`、
`semantic_audit_*`、`reflection_*`、`incumbent_updates`、
`best_effort_returned` 指标以及
`semantic_events`，用于区分 PASS、实际应用和被安全校验拒绝的神经修订。
batch summary 另汇总 `reflected_cases`、`reflection_recovered`、
`reflection_recovery_rate`、`avg_reflection_calls` 和 `best_effort_plans`。

2026-09-27 的 V3 定向回归中，`id_0、40、44、48、103` 在 6 个 solver rounds
内达到官方 100%；`id_1` 在 0 个 solver rounds、0 次路线查询后被 constraint
propagation 证明无合规酒店候选。`id_22` 已从 IR parser exception 修复为可交付，
但其 soft hotel preference 与官方 hard-constraint 口径仍不一致。

此前 V2 回归中的 `id_18、23` 也达到官方 100%。这些都是定向回归，
不是全量成绩。旧 120-case 的 12.5% 结果属于重构前版本，必须用上面的全量
命令重新测试后才能比较。

2026-09-28 的 V7 回归中，中文 `id_0..9` 内部验证并交付 9/10，平均 2.8 次
LLM、45.4 次 tools；官方 delivery、CaseAcc、composite 均为 90%，9 个交付计划
全部通过。`id_1` 因 official evidence 中没有酒店满足 `decoration_year > 2025`
而 fail-closed（10 tools、0 routes、1 solver round）。该连续十例不是随机样本，
不能替代 120-case 全量复测。

V7.2 定向回归中，`id_1` 的 hotel hard predicate 由成功且 exhaustive 的 official
hotel query 建立 `EXHAUSTIVE_NO_WITNESS` 证书，因此保持 0 次 reflection；这不再是
“调用过 query 即视为完整”的推断。`id_48` 在删除 non-convex pairwise MIQP、加入
Relax-and-Complete 和 parser 空占位结构清理后，使用 4 次 LLM、42 次 tools、0 次
reflection，官方 CaseAcc/composite 均为 100%。这些仍只是定向回归。

## id_110–119 回归（V7.2）

统一最终结果位于 `travelplanning/results/solver_certificate_110_119_final_zh/`：

```json
{
  "total_cases": 10,
  "plans_evaluated": 5,
  "delivery_rate": 0.5,
  "case_acc": 0.4,
  "composite_score": 0.45
}
```

删除 adjacent-candidate pair binaries 前，首轮 10 cases 因 size-limited Gurobi 在 6 个
case 抛异常，交付为 0。删除后不再出现该 solver exception。最终 5 个交付计划的 8 个
commonsense dimensions 全部通过；`id_110、115、117、118` 同时通过全部 personalized
constraints。`id_116` 的唯一官方失败是 evaluator 将“最好能提供机器人送餐服务”按
hard constraint 处理，而当前 IR 按明确的 soft-modality 规则处理；这是 policy mismatch，
不是时间/路线错误。

未交付主要是：长行程 selected-route acquisition 在 400 tool budget 内未收敛
（112/113）、restaurant/template coverage（114）、soft hotel predicate 导致的保守
`COVERAGE_GAP`（111），以及 119 的一次 parser 随机失败。正式比较必须固定随机性并将
parser/API failure rate 一并报告，不能用成功重跑替换失败样本。

## id_110–119 回归（V7.3）

最新统一结果位于 `travelplanning/results/solver_110_119_final_v2_zh/`：

```json
{
  "total_cases": 10,
  "plans_evaluated": 9,
  "delivery_rate": 0.9,
  "case_acc": 0.6,
  "composite_score": 0.75
}
```

本轮加入线性 route-evidence reuse priority、无法 ground 的 transport macro 淘汰、
restaurant category suffix normalization、非空 evidence merge，以及多 anchor
`nearby_locations`。9 个交付计划的全部 commonsense dimensions 均为 100%；
`id_110、113、115、117、118、119` 同时通过全部 personalized constraints。

官方 evaluator 将礼貌措辞中的可验证要求（例如“最好在某时间范围”“最好提供某服务”）
作为 binary personalized constraints。当前 benchmark policy 因此只把明确写为“非必须/
不强求/可有可无/optional”的条件放入 `soft_preferences`。若研究目标包含自然语言模态，
应将 benchmark-parity 与 language-faithful policy 作为两个独立 variant 报告。

残余失败中，`id_112` 是长行程 route-frontier 的 search-order variance（同代码定向运行曾
官方 100%）；`id_111、114、116` 已交付且 commonsense 100%，但仍分别存在 newest hotel、
train time-window、hotel service 等 personalized selection mismatch。
