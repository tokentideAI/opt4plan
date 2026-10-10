# Solver-Informed Travel Agent：全量测试诊断

更新时间：2026-09-26。分析对象为 `travelplanning/GurobiAgent/solver_informed_travel_agent.py`，中文 TravelPlanning case `id_0`–`id_119`，使用 `deepseek-flash`、`--no-react-fallback` 和项目自带的 converted-plan 评估器。本文件是对现有测试产物的诊断；本次整理文档没有修改 agent 或重新运行 API 测试。

## 1. 结果与口径

| 测试 | 样本数 | 生成行程 | 官方完全通过 | 官方 CaseAcc | 综合分 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 最终 10-case smoke test，`id_0`–`id_9` | 10 | 10 | 10 | 100% | 100% |
| 全量中文测试，`id_0`–`id_119` | 120 | 22 | 15 | 12.5% | 16.46% |

10-case 记录：`travelplanning/results/solver_informed_10_zh_complete/`。全量记录：`travelplanning/results/solver_informed_all_zh/`。每个目录的 `summary.json` 是内部推理/求解结果，`evaluation/evaluation_summary.json` 是官方评估结果，`trajectories/id_N.json` 保存该 case 的 ProblemIR、求解状态、工具调用指标及失败诊断。

`evaluated: 22` **不是** CaseAcc 的分母。全量 120 条中仅 22 条有 `converted_plans` 可评估；评估器成功处理了这 22 份文件，其中 15 份同时通过所有 commonsense 与个性化约束。因此：

- 交付率：`22/120 = 18.33%`。
- 已交付行程的完全通过率：`15/22 = 68.18%`。
- 官方全量 CaseAcc：`15/120 = 12.5%`。
- 官方 commonsense 加权分：`17.92%`；个性化约束分：`15.00%`；两者平均后的综合分：`16.46%`。这些分数同样以全部 120 条为分母，未交付的 case 计零。
- 全量推理平均每 case 约 `2.17` 次 LLM 调用、`90.66` 次工具调用；这些平均值包含失败 case，不能单独解释准确率。

先前 10/10 的外推无效：前 10 条恰好全部是 **2 天行程**，而全量数据按 2–7 天分层，每档 20 条。按天数统计如下：

| 行程天数 | 总数 | 生成行程 | 官方完全通过 |
| ---: | ---: | ---: | ---: |
| 2 | 20 | 14 | 11 |
| 3 | 20 | 7 | 4 |
| 4 | 20 | 0 | 0 |
| 5 | 20 | 1 | 0 |
| 6 | 20 | 0 | 0 |
| 7 | 20 | 0 | 0 |

这表明当前实现首先是**交付能力**的问题，尤其无法稳定处理 4 天以上的计划；对已交付行程再做精修不足以解决主要损失。

## 2. 98 条未生成行程的失败分布

`summary.json` 的最终 `last_reason` 计数如下。它只记录最终失败点，不一定等于最早的根因；例如工具参数错误可能最终表现为“无候选”。

| 最终原因 | 数量 |
| --- | ---: |
| `day MILP infeasible` | 41 |
| `no intercity transport candidate` | 25 |
| `could not form day templates` | 8 |
| `no candidates for d5:lunch` | 5 |
| `no hotel candidate` | 5 |
| `required attraction not in official recommendations` | 4 |
| `no candidates for d4:lunch` | 3 |
| `no candidates for d4:a1` | 2 |
| `verification failed` | 1 |
| `anchor route misses deadline` | 1 |
| `no candidates for d5:dinner` | 1 |
| `no candidates for d3:a1` | 1 |
| `arrival-to-hotel route unavailable` | 1 |

合并为冲突类别：时间不可行 43 条、交通无候选 25 条、模板不完整 9 条、餐厅无候选 9 条、景点无候选 7 条、酒店无候选 5 条。这里的 `temporal_infeasible` 不能直接解读为用户问题本身无解：它也可能是固定模板、候选截断或过早选定交通/酒店造成的局部不可行。

## 3. 已确认的实现缺陷与强证据

### 3.1 时间边界的类型比较错误

25 条“无城际交通候选”中，22 条的 ProblemIR 包含 `departure_time` 或 `arrival_time` 约束。典型例子 `id_18` 把“上午 6 点到 7 点出发”解析为字符串边界 `"06:00"`、`"07:00"`，而工具记录被规范化为整数分钟。`_compare(actual, op, expected)` 对 `>=`/`<=` 没有统一时间类型；整数与时间字符串比较抛出异常后返回 `False`，使候选被错误过滤。相同模式也见于 `id_23`、`id_101`、`id_104` 等。此处是代码层面可确认的 bug，但不能据此断言 22 条全部只由它导致。

修复方向：在验证 ProblemIR 或比较前，把所有时间边界统一转换为分钟；对无法解析的值返回 `UNKNOWN`/明确诊断，而不是把异常当作约束违反。增加 `06:00 <= departure_time <= 07:00`、跨午夜和日期边界单元测试。

### 3.2 官方工具 schema 与实际调用接口不完全一致

全量轨迹中有 15 次航班请求报 `FlightQueryTool.<lambda>() missing 1 required positional argument: 'seat'`。公开 schema 的 `query_flight_info` 仅要求 `origin`、`destination`、`depDate`，但实际工具调用仍要求 seat 参数；按 schema 成功编译并不保证工具能执行。另有 10 次餐厅扩展请求报缺少必需的 `latitude`。后者来自冲突扩展时仅凭城市发起 `recommend_restaurants` 请求，而该工具 schema 要求经纬度。

修复方向：为航班工具提供符合实际实现的默认舱位或在官方接口层修正协议；餐厅扩展必须先获取官方坐标，再按坐标查询。记录“schema 合法但工具执行失败”的独立指标，避免只在最终原因中看到“无候选”。

### 3.3 多日模板与固定候选池不相容

默认候选池只取 8 个景点、6 个餐厅，而 `_required_attraction_count()` 对多日游使用约 `2 × (days - 1)` 的景点密度，`_build_templates()` 又持续生成每日用餐与景点 slot。求解器跨日禁止复用已选景点/餐厅。随着天数增加，当前候选可能被用完，或者剩余候选与营业时间、路程、餐期窗口不相容；轨迹中出现 `d4:lunch`、`d5:lunch`、`d4:a1` 等候选耗尽错误。41 条 `day MILP infeasible` 也集中出现在 3–7 天，且 4 天、6 天、7 天的交付数均为零。

上述设计与分层结果构成强证据，但 41 条 MILP 失败并非全都已逐条证明是候选池耗尽；部分可能是固定活动顺序、餐厅营业时间或路线时长冲突。应对每条失败记录 slot、候选数、被排除原因及 IIS 对应的人类可读约束，再量化各子因。

修复方向：按行程长度与每日需求动态扩充候选；先检查覆盖量，再创建 slot；用 day-specific opening/closure 过滤；必要时让求解器决定每天活动数量，而不是在模板中固定。避免简单地无限增大候选池，以免相邻层的路线查询呈笛卡尔积增长。

### 3.4 过早选定单一交通/酒店，缺少真正的跨方案改选

当前流程先用 `_select_transport()`、`_select_hotel()` 各选一个方案，再构造和求解日程。即使该组合的抵达时间、位置或价格导致日程不可行，冲突扩展也没有系统地尝试次优车次/航班、另一家合规酒店、不同活动密度与顺序。`id_13`、`id_103`、`id_105` 的轨迹均在日程 MILP 阶段不可行，不能仅凭这个结果判定原始请求无解。

修复方向：把交通/酒店纳入上层候选搜索或联合求解；失败时根据冲突类型进行有边界的备选组合搜索，并区分“候选不足”“证据不足”“已证明无解”。现有 `Solve → Diagnose → Acquire → ReSolve` 在多日场景尚未实现有效的跨方案重规划。

### 3.5 ProblemIR 解析把软偏好或目标误作硬约束

抽样轨迹显示：`id_36` 的“最好能提供电视投屏”变成酒店硬性服务约束；`id_65` 的“评分最高”变成 `rating == "max"`，而非按评分最大化的目标。`id_85` 也将“最好带电视能投屏”放进硬约束。这类解析可造成“无酒店候选”，即使存在满足用户硬要求的酒店。`id_11` 的“最好是综合风味”也被写成餐厅硬性菜系要求，可能妨碍模板形成。

修复方向：ProblemIR 明确区分 `hard_constraints`、`soft_preferences`、`objective`；对最高/最便宜等比较级做结构化归一化；对“最好、尽量、优先”等语气进行校验。模板失败的 8 条还需逐条检查餐厅要求是否能映射到实际餐期，不能把它们全部归因于解析。

## 4. 已交付行程的官方评估错误

22 份已交付行程中，15 份完全通过，7 份至少一项失败。官方 `error_statistics` 给出的具体错误如下；`id_35` 同时出现两项错误，所以错误出现次数合计为 8。

| 官方检查 | 次数 | Case | 现象 |
| --- | ---: | --- | --- |
| `essential_attraction_coverage` | 3 | 15、25、35 | 返程晚于 14:00，但返程日未安排景点 |
| `train_earliest_departure_direct` | 1 | 70 | 未选到要求的去程车次 `G171` |
| `restaurant_closest_to_attraction` | 1 | 12 | 缺少木龙湖附近指定的最近餐厅 |
| `restaurant_specific_cuisine_nearby` | 1 | 10 | 缺少木龙湖附近指定菜系餐厅 |
| `seamless_intercity_transfers` | 1 | 32 | 末日城市段与实际联程第一段不匹配 |
| `attraction_type_highest_rated` | 1 | 35 | 缺少指定最高评分历史文化景点“天后宫” |

这些错误说明 agent 内部 verifier 的成功标记不等价于官方通过，尤其缺少官方的返程日最低景点覆盖、部分 superlative/nearest 的精确选择，以及联程交通的逐段表示。`id_70` 的 ProblemIR 把“最早到达”作为目标，而官方要求最早出发，值得对照原 query 核实解析是否偏离用户意图。不要用官方隐藏答案直接注入 agent；应把对应的可公开规则编入解析校验、候选选择和 verifier。

## 5. 建议的修复与验证顺序

1. **先修确定性错误**：统一时间比较；修航班 seat 与餐厅经纬度参数；补单元测试。随后只重跑受影响的 25 个交通失败 case 和相关工具错误 case，确认交付数是否上升。
2. **再修多日覆盖和调度**：按天数扩充景点/餐厅证据，约束候选图规模；输出每个 slot 的候选数量、排除原因和可读 IIS；至少分层重跑 2、3、4、5、6、7 天各一批。
3. **增加跨方案搜索**：交通、酒店、每日活动数量与顺序都要有备选；只有穷尽合规备选后才报告 `INFEASIBLE`。避免反复调用同一工具而没有新增证据。
4. **校正 ProblemIR 与 verifier**：区分软硬约束，正确处理最高/最近/最早等目标；覆盖返程日景点、联程逐段、预算和所有个性化约束。
5. **每轮同时报告四项指标**：交付率、已交付计划通过率、全量 CaseAcc、综合分，并给出按天数分层的结果。保留 `--no-react-fallback` 作为主方法消融；若开启 fallback，须单独报告其成功贡献。

结论：当前单文件 solver-informed 架构在短程 case 上已能生成高质量的、官方可验证的计划，但全量泛化的主瓶颈在**证据获取与多日候选/模板设计**，并叠加时间比较及工具参数等确定性错误。不能把前 10 条的 100% 成绩视为全量性能；以 120 条测试为准，当前 CaseAcc 是 **12.5%**。

## 6. V2 重构进展（2026-09-27）

本节记录上述诊断之后的代码状态。此前第 1–5 节的 120-case 数字属于旧实现，尚未用 V2 全量重跑，不能把下面的定向回归结果与旧全量结果直接比较。

### 6.1 已落地的架构修改

- ProblemIR 在进入 planner 后统一规范化时间、时长、数值和日期；`06:00` 等时间边界转成分钟。`after 2025` 保持严格 `> 2025`。
- transport、hotel、attraction、restaurant 明确区分 hard constraints、soft preferences 和 objective；`最好/尽量/优先` 不再自动成为硬约束，`highest/max` 被转换成优化目标。
- 固定 official tools 改为显式参数 adapter；餐厅推荐必须先获得 official coordinate，flight 空结果的 runtime 接口异常被解释为空证据而非候选。
- transport/hotel 不再各选一个后锁死，而是建立至多 24 个 macro alternatives，并先做固定成本预算下界剪枝。
- 新增 global multi-day assignment master：跨日景点/餐厅不重复、必选/top-k 景点以全局覆盖约束出现、预算下界进入 master。必选景点不再预先钉死到具体日期。
- route acquisition 改为 lazy selected-edge 查询。已查询但无结果的边会写回 edge-level no-good；日级时间失败写回 day-level no-good。
- top-k 类别要求只定义必选实体，不再把类别过滤器错误施加到所有普通观光 slot；同时获取类别候选和普通候选。
- renderer 不再把文本正则解析回 JSON；`ConvertedPlan` 直接由结构化 solution 和 evidence records 生成，文本只是 view。
- verifier 增加到达/返程日覆盖、路线事实、时间、营业时间、时长、重复实体和预算检查；内部成功仍应称为 structural verification，而非完整官方 certificate。
- 对 late-arrival empty day 做了明确处理：到达和入住晚于观光日截止时间时，酒店到酒店的零路程不会再被判成时间不可行。

### 6.2 新增离线回归

`travelplanning/GurobiAgent/tests/test_solver_informed_agent.py` 当前有 7 个无需模型 API 的测试，覆盖：

- typed time normalization 与数值比较；
- soft constraint/objective 与严格 renovation year 语义；
- `自然风光类 → 自然风光` 规范化及伪造的“行程附近”锚点清理；
- 同一请求中多个 superlative attraction groups；
- train/hotel official adapter 参数；
- restaurant coordinate fact contract；
- global cross-day entity uniqueness。

当前命令结果：

```text
python -m py_compile ...                         PASS
python -m unittest discover ...                 7/7 PASS
```

### 6.3 V2 实际 API / 官方 evaluator 回归

以下 cases 均使用 `deepseek-flash`、中文、`--no-react-fallback`，并通过项目自带官方 evaluator：

| Case | 覆盖问题 | 交付 | 官方结果 |
| ---: | --- | ---: | ---: |
| 0 | 短程基本回归、软偏好 | 是 | 100% |
| 18 | 06:00–07:00 typed time、all-matching union | 是 | 100% |
| 23 | top-1 类别实体 + 普通景点候选分离 | 是 | 100% |
| 40 | 22:25 晚到、空首日、指定/附近餐厅 | 是 | 100% |
| 103 | 7 天 global assignment、早返程、餐期窗口 | 是 | 100% |

其中 case 40 从失败时 121 次工具调用下降到成功时 39 次；case 103 在 56–60 次工具调用范围内成功，说明 V2 已能交付至少一个 7 天 case。以上是定向 regression，不代表 120-case CaseAcc。

### 6.4 尚未完成的问题

- case 65 这类 5 天、多 superlative group 的组合仍会产生较大的路径候选图。相邻路径二次目标在 size-limited Gurobi license 下必须做规模门控；回退到线性 master 后仍可能需要过多 acquire/re-solve rounds。
- exact day ordering 仍采用固定 slot order，尚未让 solver 联合决定所有活动排列；这仍可能提前排除可行顺序。
- global master 中 route cost 使用非负下界，精确 route cost 在 selected edges 返回后复核；预算冲突虽可切换 macro/assignment，但还不是完整 Benders cut。
- multi-segment intercity transport 的 structured rendering 仍需继续增强，尤其要逐段表达 transfer，而不是只依赖组合编号。
- public-rule verifier 已覆盖主要结构规则，但还未逐条证明与官方 evaluator parity。
- generic canonicalizer 仍保留兼容性解析；长期应继续拆成逐工具 canonicalizer，减少 silent alias error。

### 6.5 下一次全量测试口径

V2 全量命令见 `travelplanning/GurobiAgent/README.md`。建议使用独立输出目录 `solver_informed_v2_all_zh`，不要覆盖旧结果。全量完成后必须重新报告：

1. delivery rate；
2. delivered-plan pass rate；
3. 120-case CaseAcc；
4. composite score；
5. 2–7 天分层结果；
6. LLM/token/tool/route calls；
7. conflict kind 与 solver round 分布。

V2 当前可支持的准确结论是：确定性时间/工具协议错误已经修复，macro alternatives、global assignment 和 lazy routes 已进入实际执行路径，并在 2、3、4、7 天定向 cases 上得到官方满分；但尚不能声称整个 120-case benchmark 已被解决。

## 7. `solver_informed_all_zh` 最新全量目录诊断（2026-09-27）

分析对象：

- `travelplanning/results/solver_informed_all_zh/summary.json`
- `travelplanning/results/solver_informed_all_zh/evaluation/evaluation_summary.json`
- 同目录的 `trajectories/`、`converted_plans/` 和逐 case evaluator 结果。

### 7.1 首先修正结果口径：evaluation 目录混入旧计划

当前 `summary.json` 表明本轮实际状态是：

| 指标 | 数值 |
| --- | ---: |
| 总 case | 120 |
| 本轮成功生成 | 21 |
| 本轮失败 | 99 |
| 本轮交付率 | 17.50% |

但 evaluator 从复用的 `converted_plans/` 中找到了 36 份计划，并据此报告：

```json
{
  "evaluated": 36,
  "case_acc": 0.21666666666666667,
  "composite_score": 0.27239583333333334
}
```

多出的 15 份不是本轮 `summary.json` 中的成功 case，而是旧运行残留：

```text
id_0, id_1, id_3, id_5, id_7, id_8, id_9,
id_12, id_15, id_19, id_24, id_25, id_28, id_32, id_35
```

因此 21.67% 不能解释为一次干净运行的 CaseAcc。把 evaluator 结果与本轮 21 个成功 ID 取交集后：

| 干净重算指标 | 数值 |
| --- | ---: |
| 本轮交付 | 21/120 = 17.50% |
| 本轮官方完全通过 | 16/120 |
| 本轮干净 CaseAcc | 13.33% |
| 已交付计划完全通过率 | 16/21 = 76.19% |
| Commonsense（按 120 归一） | 17.40% |
| Personalized（按 120 归一） | 14.17% |
| Composite（按 120 归一） | 15.78% |

后续全量测试必须使用新的空输出目录，例如 `solver_informed_v2_all_zh_clean`。不能在已有目录上直接重跑并让 evaluator 扫描残留的 converted plans。

### 7.2 第一瓶颈：信息获取闭环在第 6 轮被截断

99 个本轮失败按终态划分：

| 终态 | 数量 |
| --- | ---: |
| `NEEDS_INFO` | 70 |
| `INFEASIBLE` | 27 |
| ProblemIR/agent exception | 2 |

70 个 `NEEDS_INFO` 进一步分解为：

| 最终原因 | 数量 |
| --- | ---: |
| `selected route facts required` | 66 |
| `arrival-to-hotel route required` | 4 |

66 个 selected-route case 全部具有相同特征：

- 全部恰好停止在 `solver_rounds = 6`；
- 平均工具调用 65.76 次；
- 平均路线查询 40.35 次；
- 工具错误总数为 0。

这不是“完全没有探索”，也不是 official tools 失败，而是：

\[
\boxed{
\text{lazy route acquisition 尚未收敛}
+
\text{max-solver-rounds=6 过早截断}
}
\]

失败 case 平均工具调用 51.35 次，成功 case 平均 50.57 次，数量几乎相同。因此不能简单用“失败 case 调用工具更少”解释结果。更准确的表述是：agent 已经查询很多事实，但每轮只验证少量 provisional itinerary，相似候选路径使 acquire/re-solve 收敛过慢。

### 7.3 第二瓶颈：候选、模板和约束本身不可行

27 个 `INFEASIBLE` 的主要分布：

| 冲突 | 数量 |
| --- | ---: |
| `template_incomplete` | 12 |
| `no_hotel_candidates` | 9 |
| `no_transport_candidates` | 2 |
| `no_attraction_candidates` | 2 |
| `coverage_infeasible` | 1 |
| `budget_infeasible` | 1 |

12 个 template failures 中已记录的具体原因：

| 原因 | 数量 |
| --- | ---: |
| `restaurant_requirements_unassigned` | 7 |
| 缺少细分诊断 | 3 |
| `required_attractions_exceed_slots` | 1 |
| `required_attraction_group_missing` | 1 |

7 个餐厅要求无法分配的 case 中：

- `id_119`、`id_90`、`id_94` 至少一个 requirement 的匹配候选数为 0；
- `id_16`、`id_38`、`id_47`、`id_60` 已有匹配候选，但仍无法映射到 meal slots，说明问题在餐期、营业窗口、固定顺序或日容量，而不是纯粹缺少更多餐厅。

因此提高 solver rounds 只能直接缓解 `NEEDS_INFO`，不能解决全部 27 个已判不可行 case。后者需要修改候选覆盖、活动容量、餐期匹配、macro alternatives 和 slot/order 模型。

### 7.4 ProblemIR 仍有两个确定性解析失败

| Case | 错误 |
| ---: | --- |
| 22 | `ProblemIR parse failed: 'str' object has no attribute 'get'` |
| 48 | `24:00` 被 clock validator 视为非法值 |

需要补充：

- JSON schema/type validation；
- 针对错误字段的局部 parser repair；
- `24:00` 作为闭区间结束边界的显式支持；
- 禁止把字符串、模糊词或错误容器直接送入 constraint normalizer。

### 7.5 本轮已交付计划的质量问题

本轮 21 个成功生成 case 中，16 个官方完全通过。真正属于本轮的 5 个官方失败为：

| Case | 官方失败 |
| ---: | --- |
| 70 | 未选到最早直达列车；附近指定 tag 餐厅错误 |
| 103 | 八一广场附近指定 tag 餐厅错误 |
| 10 | 木龙湖附近指定菜系餐厅错误 |
| 44 | 未选择宽窄巷子附近最近餐厅 |
| 82 | 用餐时间不在餐厅营业时间内 |

其中 4 个主要属于精确选择语义：`earliest`、`closest`、指定 `tag/cuisine nearby`。它们需要进入 deterministic candidate selection、hard coverage/objective 和 verifier，而不是依赖扩大普通候选池后自然命中。`id_82` 则说明内部 opening-hours verification 与官方规则仍未完全对齐。

旧残留计划中的 `id_12、15、25、32、35` 也被 evaluator 判失败，但不应计入本轮错误率。

### 7.6 必须增加前置 Fast Feasibility / Lack Check

当前流程在证明候选覆盖之前就可能开始几十次精确路线查询。建议改为：

```text
Parse
  → Acquire Core Facts
  → Fast Feasibility / Lack Check
  → Macro Planning
  → Global Assignment
  → Lazy Selected-Edge Routes
  → Exact Scheduling
  → Verification
```

FastCheck 返回三态：

```text
PROVISIONALLY_FEASIBLE
LACK
INFEASIBLE
```

建议数据结构：

```python
@dataclass
class FastCheckResult:
    status: str
    missing: list[AcquisitionRequest]
    conflict_kind: str | None
    diagnostics: dict
```

FastCheck 至少检查：

1. ProblemIR 字段类型、日期、时间、人数和房间数是否合法；
2. outbound、return 和 hotel 是否各有合规候选；
3. transport + hotel 最低固定成本是否已经超过预算；
4. named、top-k、all-matching attractions 是否有足够实体和可用 slots；
5. 每个 nearby/tag/cuisine restaurant requirement 是否有匹配候选；
6. lunch 与 dinner 是否分别有满足营业窗口的候选，不能只看餐厅总数；
7. 每日最小活动时长与可用时间容量是否矛盾；
8. price、opening/closing time、duration、coordinates 等 FactContract 是否齐全；
9. return-day attraction、餐期和必选活动的粗粒度覆盖是否可能；
10. 是否应补事实、扩候选、换 macro，还是已经能证明不可行。

关键语义必须区分：

```text
餐厅详情尚未查询
→ LACK(required fact)

指定坐标附近的官方餐厅已经全部返回，但没有任何一家满足 tag
→ INFEASIBLE for this requirement/candidate source

当前 hotel macro 不可行，但仍存在其他合规酒店
→ CHANGE_MACRO

所有 macro 的固定成本下界均超过预算
→ PROVEN_INFEASIBLE
```

核心原则：

\[
\boxed{
\text{先验证候选覆盖和粗粒度容量，}
\text{再支付精确路线查询成本}
}
\]

FastCheck 不替代 exact route solver。它负责尽早发现缺失或明显不可行；最终地理、时间和成本证据仍必须来自 official route tools。

### 7.7 下一轮实验设计

首先用全新输出目录执行相同代码的 solver-round ablation：

```text
max_solver_rounds = 6 / 12 / 24 / 64
```

固定 model、prompt、workers、tool-call budget 和 case 集合，比较：

1. `NEEDS_INFO` 数量；
2. delivery rate；
3. delivered-plan pass rate；
4. 全量 CaseAcc/composite；
5. 平均 route queries 与 solver rounds；
6. 2–7 天分层交付率；
7. `NEEDS_INFO → success` 与 `NEEDS_INFO → INFEASIBLE` 的转移。

如果增加到 64 轮后 70 个 `NEEDS_INFO` 大量转为成功，则主要瓶颈确实是 acquisition depth；如果主要转成 template/temporal/coverage infeasible，则说明路线查询此前只是遮住了更深层的建模问题。

### 7.8 当前综合结论

\[
\boxed{
\text{第一瓶颈：信息获取闭环在 6 轮时被过早截断}
}
\]

\[
\boxed{
\text{第二瓶颈：缺少路线查询前的 Fast Feasibility/Lack Check}
}
\]

\[
\boxed{
\text{第三瓶颈：固定模板、精确目标语义和 verifier parity 不完整}
}
\]

因此当前问题不能概括为“检索数量不够”。更准确的结论是：检索闭环深度不足、检索前缺少快速缺失/可行性筛查，并且一部分失败属于候选和调度建模错误，增加无差别检索不会解决。

## 8. V3 Symbolic Constraint / Frontier 改进（2026-09-27）

根据 ChinaTravel symbolic reasoning 的可迁移思想，当前 agent 已加入以下非 case-specific 改进：

- `ConstraintIR`：predicate、arg-extreme、candidate coverage、restaurant requirement、budget lower bound、slot capacity 和 implication 成为类型化对象；
- `ConstraintEvaluation`：统一输出 `SAT/UNSAT/UNKNOWN`、missing facts、lower bound、diagnostics 和来源约束；
- `IncrementalConstraintEngine`：在 selected-route acquisition 前执行 constraint propagation；
- IR type checker：非 object constraint/requirement 会触发局部 LLM repair，而不是运行时属性错误；
- `24:00` 现在作为合法区间终点规范化为 1440；
- frontier acquisition：同轮构造多个 provisional assignments，统计共享缺失事实并按出现次数/优先级批量查询；
- `TRAVEL_FRONTIER_WIDTH` 和 `TRAVEL_FACT_BATCH` 控制 frontier 与单轮事实成本；
- closest restaurant 的 hard `ArgMin(route_distance)` 会要求所有候选的 official route facts 齐备，再认证最优者；
- symbolic metrics 已加入 trajectory：propagation 次数、FALSE/UNKNOWN 数、FastCheck 请求、frontier states 和 shared fact requests。

新增离线测试后当前为 11/11 通过。

### 8.1 六轮 smoke test

在 `max_solver_rounds=6`、`--no-react-fallback` 下：

| Case | 结果 | 官方分数 | 说明 |
| ---: | --- | ---: | --- |
| 0 | 成功 | 100% | 旧全量同配置未交付 |
| 40 | 成功 | 100% | 晚到空首日 + 指定餐厅 |
| 103 | 成功 | 100% | 7 天计划在 6 轮内完成 |
| 44 | 成功 | 100% | closest restaurant 从旧官方失败修复 |
| 48 | 成功 | 100% | 从 `24:00` parser exception 修复 |

`id_1` 在 10 次工具调用、0 次路线查询、0 个 solver rounds 后被 symbolic propagation 判定为 `hotel.coverage = UNSAT`，说明 FastCheck 已能在路线检索前阻止明显不可行分支。

`id_22` 从 IR parser exception 变为可交付且 commonsense 100%，但 personalized 失败：用户使用“最好有游泳池”的 soft modality，当前 IR 按 soft preference 处理，而官方 evaluator 按 hard hotel requirement 处理。这个差异应作为 benchmark semantics adapter 或评估口径问题显式报告，不应悄悄将所有“最好”改成 hard。

`id_70` 仍有两个官方个性化失败，其中“最早到达”被 evaluator 检查为“最早出发”，同样属于用户语义与官方规则不一致；“最好有等位区”也被官方当成 hard。当前 agent 不对这两个矛盾做 case-ID patch。

### 8.2 当前收益与代价

frontier batching 显著提高了 6 轮内完成概率，但会增加同轮路线查询。为避免把准确率提升简单换成无界工具成本，frontier 现采用共享次数排序和单轮 batch 上限；下一次全量实验必须同时报告 CaseAcc 与 route/tool calls，并对 `frontier_width/fact_batch` 做消融。

重复运行 `id_103` 时曾观察到同一 query 的 ProblemIR 方差：一次遗漏了回程 `duration/min` 和八一广场餐厅的 `rating/max` objective，导致官方 personalized 从 100% 降至 0。现已加入可审计的 semantic completeness repair：它只从原始 source text 恢复明确的“用时尽量短/最短”“评分高优先/最高”“最便宜”“最近”等 comparative objective，不修改实体事实或生成行程。修复后 `id_103` 再次在 6 轮内交付，95 次工具调用，官方 100%。

当前 V3 仍不是完整的 immutable `SearchState + dominance` Beam Search，也没有把所有 verifier 规则统一编译自 Constraint IR。下一阶段优先级应是：

1. 将 solution-level verifier 也编译自同一 Constraint IR；
2. 为 frontier state 增加稳定 signature、dominance 和 cost/unknown lower-bound score；
3. 从 whole-assignment no-good 继续下沉到最小局部 conflict；
4. 继续减少初始 queue 的 retrieve-all details 行为，真正让 FactContract 按需驱动 acquisition；
5. 用全新目录运行干净的 120-case V3 测试。

## 9. V3.1 Event-Triggered LLM 改进（2026-09-27）

### 9.1 客观判断

现有 V3 正常路径平均约 1 次 LLM 调用，而本地 ReAct 10-case trajectory 平均
4.8 次 LLM、22.9 个 tool results；V3 smoke 平均 1 次 LLM、89 次 tools。
因此 V3 的 LLM/tool 反馈密度约为 ReAct 的十九分之一。调用总量低本身不是错误，
但所有神经语义判断集中在第一次 `NL → IR`，会使一个“类型合法但语义错误”的
IR 被后续 symbolic/solver 精确执行。

采用的改进不是每轮调用 LLM，而是：

```text
Parse (1)
  → Semantic Audit (1)
  → deterministic propagation / tools / solver
  → conflict 时统一 Semantic Recovery (0–1)
```

暂不加入 Search Stagnation Critic。当前 stagnation 主要由路线事实、frontier 和
no-good 状态决定，尚无证据说明 LLM meta-controller 比 deterministic progress
measure 更有效；过早加入会混淆 Semantic Audit 的独立收益。

### 9.2 安全边界

Audit/Repair 只产生候选 IR，必须通过以下检查才应用：

1. origin/destination/date/travelers/rooms/budget/currency 不可变化；
2. outbound/return allowed modes 不可变化；
3. 新增 constraint/objective 的 `source_text` 必须能回指原始 query；
4. 明确含“必须/一定/务必/must/required”的来源不可被删除；
5. 新增景点、餐厅或 nearby anchor 必须出现在原请求；
6. 修订后重新执行完整 schema/type/semantic normalization；
7. no-op、非法 JSON 和越权修订全部 fail closed，继续使用原 IR。

Semantic Recovery 不是对所有 `UNSAT` 都调用。只有相关约束来源含“最好、尽量、
优先、希望、附近、左右”等可能存在 modality/reference 歧义的表达，或者来源本身
无法回指 query，才触发一次。确定性的硬约束无候选仍直接判定 infeasible。

### 9.3 新增观测与消融

trajectory 新增：

```text
semantic_events
semantic_audit_calls / repairs / rejected
semantic_recovery_calls / ir_repairs_applied / recovery_rejected
recall_expansions_proposed / accepted
```

建议用相同 case 集合和全新输出目录比较：

```text
S0: --no-semantic-audit --no-semantic-recovery
S1: semantic audit only (--no-semantic-recovery)
S2: semantic audit + conditional recovery (default)
S3: S2 + ReAct fallback
```

报告 CaseAcc、DeliveryRate、LLM calls/tokens、tool calls、solver rounds，以及
`successful cases / LLM calls`。实现后的离线测试为 16/16 通过；尚未执行干净的
120-case V3.1，因此不能提前声称全量准确率提升。

### 9.4 定向在线验证

`id_1` 验证了 fail-closed 路径：Semantic Audit 曾尝试加入 schema 不支持的字段，
field whitelist 将修订拒绝并保留原 IR；随后 FastCheck 用 10 次 tools、0 次 route、
0 个 solver rounds 证明 hotel hard constraint 无候选，没有因 LLM 建议而软化要求。

`id_48` 验证了成功路径。额外修复了两个被 Audit 暴露的通用问题：

1. 必选景点不在按类型返回的 recommendation pool 时，直接通过官方 detail tool
   定向获取，而不是错误判为 absent；
2. named must-visits 与一个 filtered top-k 请求编译成两个 selection groups，类型
   filter 不再错误施加到所有必选景点；模糊的“性价比高”不会被编译成 rating/max。

最终 `id_48` 在 3 次 LLM、63 次 tools、41 次 route queries、5 个 solver rounds
内交付，官方 commonsense 100%、personalized 100%、CaseAcc 100%。这是单 case
regression，只证明新路径没有破坏该成功样本，不代表全量提升。

在线运行还显示 auditor 偶尔会返回 schema 外字段或缺失 operator。当前处理是：
parser 最多进行三次 checker-guided 局部修复；Semantic Audit 的修订若被 validator
拒绝，最多再给一次精确 validation error 后重试，仍失败则安全退回原 IR。

## 10. V3.2 Unified Acquisition / Semantic Recovery（2026-09-27）

V3.1 的 conflict-time constraint repair 与新增 recall expansion 本质上都是失败后的
恢复动作，因此 V3.2 删除独立 repair 分支，统一为：

```text
Symbolic/Solver Conflict
  → Semantic Recovery
  → KEEP | REPAIR_IR | EXPAND_RECALL
```

没有新增 `RecallAgent/RecallState/RecallPlan/RecallMemory`。LLM recall 不是另一套
retrieval subsystem，而只是现有 acquisition queue 的受限 proposal source。

### 10.1 唯一请求通道

initial requests、constraint missing facts、solver missing facts、deterministic expansion
和 LLM recall 现在全部进入同一个 `_run_acquisition_queue()`：

```text
A = A_initial ∪ A_constraint ∪ A_solver ∪ A_LLM
  → dedup(key)
  → priority
  → official tool adapter
  → EvidenceStore
```

旧 `_acquire_essentials()` 以及主循环中重复的 request/detail execution loops 已删除。
route/location 等确定性缺失事实仍直接生成请求，不触发 LLM。

### 10.2 Recall 的安全边界

Recovery 只接受三种 intent：

```text
additional_anchor
attraction_type
unrestricted_attractions
```

其中 additional anchor 必须已经存在于 EvidenceStore 且具有官方坐标；attraction
type 必须属于 public tool schema 列出的六类。LLM不能输出餐厅/景点答案、路线、
时间或价格。所有合法 intent 被 deterministic compiler 转成现有
`AcquisitionRequest`，再走完全相同的工具路径。

每个 recall-sensitive constraint 最多接受一次 recall expansion：

```python
self._recall_attempted: set[str]
```

同时以 conflict signature 记录 `_recovery_attempted`，防止 KEEP/无效扩展循环调用。
当前 recall-sensitive 范围仅包括 `attraction_coverage` 和
`restaurant_requirement`；budget、route missing、transport/hotel hard infeasibility
不会用 recall 掩盖。

### 10.3 指标与验证

新增指标：

```text
semantic_recovery_calls
ir_repairs_applied
recovery_rejected
recall_expansions_proposed
recall_expansions_accepted
```

CLI 使用 `--no-semantic-recovery`；旧 `--no-constraint-repair` 仅作为兼容 alias。
离线测试现为 18/18，通过内容新增：虚构 anchor/type 不会编译成请求，以及同一
constraint 的第二次 recall 会被拒绝。该结构重构后尚未完成新的全量 120-case
实验，不能把代码路径简化直接解释为准确率提升。

V3.2 在线回归 `id_48`：3 次 LLM、58 次 tools、34 次 route queries、4 个 solver
rounds，官方 commonsense/personalized/CaseAcc/composite 全部为 100%。相较此前同
case 的 63 tools、41 routes、5 rounds 更低，但由于 LLM parse/audit 存在随机性，
单 case 差异不能归因于 queue 重构；这里只能确认成功路径未被破坏。

## 11. V7 减法重构（2026-09-28）

本轮目标不是增加规划能力，而是删除重复求解与重复语义解释，使实验归因更清楚。

### 11.1 Solver 收敛

Global master 已经把每个 slot 限制为唯一 candidate，因此原 day-level Gurobi 的
binary variables 恒等于 1。本轮删除第二层 candidate MILP 及无完整意义的 greedy
fallback，改为固定链上的 earliest-time propagation：

```text
Global MILP entity assignment
  → selected-edge official routes
  → deterministic earliest-time schedule
  → one structural verification
```

若最早可达安排已违反 closing time、slot latest 或 day deadline，则所有更晚安排也
不可行；这时 deterministic scheduler 能给出与第二层 MILP 等价的固定顺序可行性结论。

### 11.2 语义职责收敛

validator 现在只负责 schema、field/operator whitelist、日期/时间/时长/数值和 enum
规范化。删除了以下 query-regex 或 source-marker 修补：

- 从“最好/尽量/优先”自动搬移 hard/soft；
- 从原 query 补 highest/closest/shortest objective；
- 从 query 正则制造额外 superlative attraction groups；
- 把模糊 nearby anchor 静默改成 null；
- 把 `rating == "max"` 静默改成 objective。

最后一类现在直接产生 validation error，交由 parser 的 checker-guided retry 修复。
语义等价性统一由 Semantic Audit 与受限 Semantic Recovery 处理。

### 11.3 主控制流与输出收敛

- `_attempt_solve()` 不再内部 verify，controller 只验证一次；
- 删除进入 solver loop 前的重复 propagation；
- follow-up acquisition 按触发 tool 类型生成，不再每次请求后扫描所有 followups；
- 删除主文件内置 ReAct fallback 与相关 CLI/metrics；
- `ConvertedPlan` 只构造一次，文本仅是 JSON artifact 的 view；
- batch 输出区分 `produced_plan`、`structured_ready`、`internal_verified`、
  `official_case_acc`，兼容字段 `success = internal_verified`；
- flight runtime exception 不再伪装为空候选，而记录 `TOOL_PROTOCOL_ERROR`；
- `_compare()` 的类型错误返回 `UNKNOWN`，constraint aggregate 会保留 hard UNKNOWN。

本轮还删除 dead selectors、unused state/parameters 和无实际检查能力的象征性
`slot_capacity` / return-day implication constraints。后续若重新加入，必须具有真实的
partial-state evaluator，不能直接返回 SAT/UNKNOWN 占位。

代码从重构前约 4593 行降至约 4091 行。该数字不是质量指标，只说明重复执行路径已
实质删除。离线回归增加 deterministic scheduler 的等待与 closing-time infeasibility
测试，目前 21/21 通过。

### 11.4 在线回归结果

第一次 `id_48` 回归发现一个有价值的 schema gap：`selection=named` 同时带有 scoped
category constraint/objective 时，类别 filter 被误施加到所有普通景点槽，只召回 3 个
景点并导致长行程候选耗尽。修复没有恢复 query regex，而是加入纯 typed-IR 结构归一化：

```text
named required_names + scoped constraints/objective
  → named coverage + independent required_group(count=1)
  → unrestricted official recall for ordinary slots
```

修复后 `id_48` 为 4 次 LLM、56 次 tools、34 次 routes、4 个 solver rounds，官方
CaseAcc/composite 均为 100%。

随后运行中文 `id_0..9`：内部 verified/交付 9/10，平均 2.8 次 LLM、45.4 次 tools；
官方 evaluator 的 delivery、CaseAcc、composite 均为 90%。9 个交付计划全部通过所有
commonsense 与 personalized checks。唯一未交付的 `id_1` 在 10 次 tools、0 次 routes、
1 个 solver round 后被 symbolic FastCheck 判定为无酒店满足硬约束
`decoration_year > 2025`，属于 fail-closed，而非生成了错误计划。

这组 10-case 结果验证了 V7 删除重复 day MILP 后没有损害这些样本的时间可行性；但它
不是随机抽样，也不能替代 120-case 全量复测。

## 12. V7.1 Conflict-Guided Reflection（2026-09-28）

Reflection 被实现为 acquisition controller 的受限恢复动作，而不是自由式 Reflexion：

```text
Solve / Verify
  → deterministic missing facts / expansion / no-good
  → no progress
  → conflict reflection (max 2)
  → grounded retrieval intent
  → AcquisitionRequest queue
  → ReSolve
```

状态语义新增 `COVERAGE_GAP`、`SEARCH_EXHAUSTED` 和 `PROVEN_INFEASIBLE`。
本节原实现曾把部分 hard `UNSAT` 直接升级为 `PROVEN_INFEASIBLE`；V7.2 已废止该
推断，只有通过 deterministic certificate verifier 的 conflict 才能使用该状态。

Reflection 只能输出：

```text
KEEP
RECHECK_SEMANTICS
EXPAND_RECALL
```

其中 retrieval intent 仍只能使用已 grounded 的 hotel/attraction anchor 或 public tool
schema 中的六类 attraction type。LLM 不得返回餐厅/景点实体、路线、价格或 itinerary。
每个动作以 status、conflict、action 和 compiled request keys 构造 signature；重复动作
会被拒绝，最多调用 `--max-reflections` 次，默认 2。

Planner 现在保留 verification 后的 best incumbent。失败 assignment 会加入 exact
no-good，迫使下一轮搜索其他组合；若 reflection budget 用尽仍无 exact solution，但存在
可转换 incumbent，则输出 best-effort ConvertedPlan，同时保持：

```text
produced_plan = true
internal_verified = false
success = false
best_effort_returned = true
```

这允许官方 evaluator 测量 composite，而不会把 best-effort 错报为 exact success。

离线测试为 23/23，包含 best-effort 交付但不误标 exact success 的契约测试。在线触发验证：

- `id_1` 当时由旧分类器标记为 proven infeasible；该表述已由 V7.2 的证书回归取代；
- `id_26` 触发 2 次 reflection，其中 1 次受 source-grounding 检查后应用
  `RECHECK_SEMANTICS`，随后对无新合法 retrieval action 返回 KEEP；该 case 仍未交付，
  说明 reflection 不会为了交付而放宽硬约束；
- `id_48` exact path 不触发 reflection，4 次 LLM、57 tools，官方 CaseAcc/composite
  仍为 100%。

这些验证证明 trigger、安全边界和非退化路径按设计工作；尚不能证明 reflection 提升
全量准确率。需要用 `--no-reflection` 对照运行并报告 Reflection Recovery Rate、额外
LLM/tool cost、CaseAcc 和 Composite 的差异。

## 13. V7.2 Certificate-Based Failure Semantics（2026-09-28）

### 13.1 审计结论

V7.1 的最大正确性风险不是 reflection prompt，而是把“当前检索域无解”误称为
“原问题已证明无解”。以下条件现在全部不再构成全局 proof：

- recommendation 结果中没有满足者；
- candidate pool / macro generator 为空；
- `3 * days` 等模板容量不足；
- Gurobi 在当前候选域不可行；
- route frontier 或 deterministic recall policy 耗尽；
- tool error 或只发生过一次 query。

这些状态分别保留为 `UNKNOWN`、`COVERAGE_GAP`、`SEARCH_EXHAUSTED` 或 model/search
conflict，并允许 deterministic recovery、Relax-and-Complete 或 reflection 继续工作。

### 13.2 Query outcome 与完备性

`queried_keys` 只用于去重，不再表示 domain complete。每次调用另存：

```text
QueryOutcome =
  SUCCESS_NONEMPTY | SUCCESS_EMPTY | ERROR | PARSE_ERROR
  + exhaustive
  + record_count
  + error
```

只有成功且由 public protocol contract 定义为 exhaustive 的调用，才能进入证书。
`recommend_attractions`、`recommend_restaurants` 和 `search_location` 不作为完备域证据；
tool/adapter/runtime error 永远不等价于 empty result。

### 13.3 InfeasibilityCertificate

`PROVEN_INFEASIBLE` 现在必须携带 `InfeasibilityCertificate`，controller 会 fail-closed
重放检查：证据 key 必须存在、成功、exhaustive；constraint 必须仍为 hard `UNSAT`；
constraint engine 重新计算出的 certificate 必须与待验证 certificate 完全一致。

目前只允许三类 certificate：

```text
EXHAUSTIVE_NO_WITNESS
EXACT_ENTITY_ABSENCE
BUDGET_LOWER_BOUND
```

其中 budget proof 只有在 outbound、return、hotel 三个 fixed-cost domain 都完整时才
成立。名称 `EXHAUSTIVE_NO_WITNESS` 特意区别于 raw empty：例如 exhaustive hotel query
可返回 20 条记录，但没有一条满足 hard predicate，这仍是“完整域无 witness”，不是
“工具返回空”。LLM reflection action space 中彻底没有 `PROVEN_INFEASIBLE`。

### 13.4 Runtime recovery 收敛

原 `Semantic Recovery` 与 `Reflection` 合并为一个 conflict reflection controller，动作
只有：

```text
KEEP | RECHECK_SEMANTICS | EXPAND_RECALL
```

删除了实际只是 `EXPAND_RECALL` 别名的 `CHANGE_SEARCH_FOCUS`。确定性缺 route/detail
不调用 LLM；selected-route frontier 停滞时先选择 unknown edges 最少的 provisional
assignment，一次性补齐其全部缺边（Relax-and-Complete），只有候选覆盖或语义空间缺陷
才进入 reflection。

### 13.5 Solver 与 incumbent

Global master 删除了仅用于近似地理优先级的 `x[left] * x[right]` 和
`NonConvex=2`，恢复为线性 MILP；exact routes 仍在选定 assignment 后由 official tool
获取并由 deterministic scheduler 校验。best incumbent 不再使用
`evidence_coverage - 0.1 * errors`，而采用 verifier 的 8 个 commonsense 维度与估计
personalized score 形成 estimated composite。该分数仅用于内部 incumbent 排序，不声称
等同官方 evaluator。

### 13.6 其他可靠性修复与验证

- 每个 rerun case 开始先删除该 ID 的旧 trajectory/report/converted plan，避免历史污染；
- parser 丢弃 field/op/source/value 全空的 schema placeholder；对 categorical field 缺失
  operator 时只做 typed canonicalization（presence → `contains`，category → `==`），
  numeric operator 缺失仍报错；
- 每个 result directory 应保持独立，正式全量实验仍建议使用全新 output directory；
- 离线 `py_compile`、`git diff --check` 与 28/28 unittest 通过。

在线定向回归：

- `id_1`：4 次 LLM（parser/audit 各发生一次受检重试）、10 次 tools、0 次 reflection；
  official hotel query 成功返回完整
  20 条记录，但没有酒店满足 `decoration_year > 2025`，产生并通过
  `EXHAUSTIVE_NO_WITNESS` 证书。因此现在才可以审计性地称为 proven infeasible。
- `id_48`：4 次 LLM、42 次 tools、0 次 reflection，官方 commonsense、personalized、
  CaseAcc、composite 全部 100%。一次独立运行曾因模型连续三次输出空 operator 而在
  parser 阶段失败；空 placeholder 清理后成功，但这也说明正式实验应报告 parse failure
  rate，而不能只选择成功重跑。

本轮结论只证明 failure classification、certificate authority 与 exact 成功路径按设计
工作，尚未证明 Reflection 或 Relax-and-Complete 能提升 120-case CaseAcc。下一步必须
在相同模型、温度、case 集和预算下做 `--no-reflection` A/B，并同时报告 delivery、
CaseAcc、composite、parser failure、certificate count/rejection、Reflection Recovery Rate、
LLM/tool/route calls。

## 14. id_110–119 长行程回归与修复（2026-09-28）

### 14.1 首轮失败不是算法结果

首次运行 110–119 得到 0/10 交付，其中 6 个是：

```text
Model too large for size-limited license
```

静态检查发现 master 虽已删除 non-convex `x_i*x_j`，但后来又为每个未知相邻候选对
建立 `z_ij = x_i AND x_j`。长行程每层候选较多时变量数仍为
`O(|C_i||C_j|)`，因此只是换成线性形式，并没有消除模型爆炸。

最终删除全部 unknown-edge pair binaries。Global MILP 只选择实体；exact edge 继续由
selected-plan Lazy Route / Relax-and-Complete 获取。删除后 110–119 不再出现 Gurobi
license exception。

### 14.2 数据与 parser 修复

- `_service_duration()` 原先使用 `min_duration or 120`，把合法的 `0` 小时下界误当成
  missing，导致 4 个计划固定输出 2 小时并违反 evaluator maximum。现在显式区分
  `None` 与 `0`，并将正时长限制在 official min/max interval；
- transport IR 增加 public flight fact `manufacturer`，支持空客/波音 + cheapest direct；
- numeric field 禁止“合适/高”等形容词充当数值；
- categorical missing-op 可做无语义歧义的 typed operator canonicalization；
- transport soft witness 先缩小 domain，再在该 domain 内计算显式 argmin/argmax，修复
  “最便宜动车”被解释成“所有直达车中最便宜”的 scope 错误；
- certificate verifier 同时检查 source phrase 在原 query 中的邻近 modality；coverage
  certificate 也继承相关 predicate 的 modality uncertainty，避免绕过检查。

### 14.3 最终统一结果

结果目录：

```text
travelplanning/results/solver_certificate_110_119_final_zh/
```

官方汇总：

```json
{
  "total_cases": 10,
  "plans_evaluated": 5,
  "delivery_rate": 0.5,
  "case_acc": 0.4,
  "composite_score": 0.45
}
```

5 个交付计划的全部 commonsense dimensions 均为 100%。`id_110、115、117、118` 的
personalized constraints 也全部通过，因此 CaseAcc 为 4/10。`id_116` commonsense 为
100%，但 personalized 失败：官方 evaluator 将“最好能提供机器人送餐服务”作为 hard，
当前 IR 按自然语言 modality 作为 soft。这是 evaluator-policy mismatch，不应偷偷通过
regex 将所有“最好”改成 hard；若论文主指标要求官方 parity，应增加显式 benchmark
semantics adapter，并与 language-faithful variant 分开报告。

未交付诊断：

- `id_112、113`：7 天行程的 selected-route facts 在 400 tool budget 内仍未收敛；
- `id_114`：指定餐厅要求无法匹配固定 meal/template coverage；
- `id_111`：疑似 soft hotel predicate 的无 witness 不再被误证为 infeasible，保守返回
  `COVERAGE_GAP`；
- `id_119`：最终统一运行发生 parser 随机失败；另一次重试又遇到 API connection error，
  因此不能用选择性成功重跑替代该失败记录。

本组从首轮 0% CaseAcc/0% composite 提升到 40%/45%，主要归因于删除 pairwise master
变量和修复 visit duration，而不是 reflection：最终 `reflection_recovery_rate=0.2`，仍需
正式 `--no-reflection` A/B 才能判断增益。平均 141.2 tool calls/case 仍过高，下一轮 P0
应是 route acquisition 收敛与固定模板解除，而不是增加 LLM reflection 次数。

## 15. id_110–119 第二轮改进与最终复测（2026-09-28）

本轮进一步修复了四类会错误缩小搜索空间的问题：

1. Global master 增加线性的 `route_reuse_priority`，优先复用已取得的 official edges，
   不重新引入 `O(|C_i||C_j|)` pair variables；定向运行中 id_112/id_113 均曾恢复 exact。
2. restaurant category 做语义后缀归一化（如“清真菜馆”与“清真菜”），detail merge 不再
   用空 tags/services 覆盖 recommendation evidence。
3. 同一家餐厅可由多个 official anchor 推荐，state 改为累积 `nearby_locations`，solver
   和 verifier 统一按 anchor 集合判断；id_119 单独复测与最终十例运行均官方 100%。
4. exact `search_location` 为空后只淘汰无法 ground 的 transport macro，不把它升级为
   global infeasibility；这使 id_114 能转向有官方坐标的车站并构造完整计划。

此外，官方 evaluator 已通过 111/114/116 多次证明确认可验证的“最好……”条件仍属于
binary personalized constraints。因此 benchmark 运行采用显式 parity policy：只有用户
明确说“非必须/不强求/可有可无/optional”才编译为 soft；礼貌措辞“最好/希望/prefer”
不会自动放弃可验证条件。该 policy 应与 language-faithful variant 分开做 ablation。

最终统一目录：

```text
travelplanning/results/solver_110_119_final_v2_zh/
```

官方结果：

```json
{
  "total_cases": 10,
  "plans_evaluated": 9,
  "delivery_rate": 0.9,
  "case_acc": 0.6,
  "composite_score": 0.75
}
```

相对上一统一版本（0.5 delivery / 0.4 CaseAcc / 0.45 composite），分别提高
`+0.40 / +0.20 / +0.30`。9 个交付计划全部通过 8 个 commonsense dimensions；
完整通过的是 id_110、113、115、117、118、119。id_111、114、116 仅剩 personalized
selection/service mismatch，commonsense 均为 100%。id_112 在该次统一运行仍因 7 天
selected-route frontier 在 64 rounds 内未收敛而未交付；它在同代码的定向运行中曾官方
100%，说明当前主要残余是 search-order variance，而非候选或 verifier 必然无解。

本轮最终离线验证为 33/33 tests、`py_compile` 通过。Reflection recovery 在该十例仍无
可归因增益；主要提升来自 evidence preservation、groundability pruning 与线性 route
reuse。因此下一步应优先稳定 id_112 式长程 route convergence，以及把 newest/cheapest/
nearest 等 evaluator selection semantics完全纳入统一 Constraint IR，而不是增加 LLM 次数。
