以 skeptical auditor 的标准，这版已经比上一版强很多：你修掉了 `per_product` schema、统一了 LLM call/token accounting、fallback 改成 fresh context、Gurobi 加了 exact optimality gap 与 exact-plan cross-check。这些都是实质性改进。 

但我仍然不会把 `shopping_agent(5).py` 当作最终 paper artifact。现在最主要的风险已经从“solver 写错”转成了：

$$
\boxed{
\text{Retrieval / IR semantics}
>
\text{Solver}
>
\text{Engineering cleanliness}
}
$$

尤其你前面发现 **Total matched products 偏低**，从当前实现看是有明确原因的。

| 优先级    | 当前问题                                             | 审计判断                                                 | 建议                                                   |
| ------ | ------------------------------------------------ | ---------------------------------------------------- | ---------------------------------------------------- |
| **P0** | `search_products` 存在但主 retrieval 根本没用            | 很可能直接伤 Product Match                                 | 增加 exact-name/category retrieval；不要只扫 price frontier |
| **P0** | 只有少数字段能 push-down filter                         | rating distribution / season / demographic 等导致大量无关候选 | 扩展 deterministic filter compiler                     |
| **P0** | profile defaults 对非服饰商品也可能制造 `UNKNOWN`           | 可能误杀 GT                                              | profile size 只应用到明确 size-relevant slot               |
| **P0** | Gurobi cross-check 默认开启                          | 你实际上每次都在跑 Gurobi **+ DP oracle**                     | production 默认关闭，validation 单独开启                      |
| **P1** | L2/L3 强制 exhaustive candidate set                | benchmark 可跑，scalability 很弱                          | 改为 solver-guided progressive expansion               |
| **P1** | infeasible 后允许 LLM 重写 IR                         | 有 catalog-feedback / benchmark adaptation 风险         | 作为显式 ablation，不要藏在主方法                                |
| **P1** | product-ID lexicographic tiebreak                | 无用户语义，可能 benchmark overfit                           | 用官方 stable rank，或报告 equivalent-optimum               |
| **P1** | Level-3 仍先枚举 coupon plans                        | Gurobi 外面套指数级枚举                                      | coupon frontier 用 DP / Pareto compression            |
| **P2** | `name` verification 拼接 brand/color/season/gender | 语义边界模糊，可能 false SAT                                  | parser 正确拆字段，name 只验证 name/category                  |
| **P2** | size regex 写死                                    | 一位数鞋码等可能直接非法                                         | 用 catalog vocabulary，而不是 regex                       |
| **P2** | 一个 2954 行 monolith                               | production/oracle/fallback/infra 混在一起                | 拆模块                                                  |
| **P3** | compatibility / logging code 很多                  | 不影响分数，但妨碍维护                                          | 单独 adapter / trace layer                             |

下面是我认为最关键的几个问题。

### 1. 当前最大的算法问题：你仍然没有真正解决 retrieval recall

`READ_ONLY_TOOL_NAMES` 里虽然有 `search_products`，但 `_tool_filter_for_constraint()` 实际只把 brand、color、size，以及部分 numeric fields 转成 filter。`name`、`suitable_season`、`target_demographic`、五档 rating distribution 都没有 retrieval push-down。

然后 `_certify_slot_coverage()` 基本做：

$$
\text{available filters}
\rightarrow
\text{sort by price}
\rightarrow
\text{inspect prefix}.
$$

并且每 slot 默认最多检查 200 个。

所以一个 slot 如果主要靠：

```text
name = "Essential Logo Crewneck Tee"
1-star < 30
season = all seasons
```

来定位商品，当前系统可能没有先用最重要的 **name/type anchor** 缩小集合。

这和你之前 Total matched products 偏低高度一致。

但我不建议直接把 semantic `search_products` 当作 exact coverage source，因为 top-k semantic search 通常不能证明：

$$
C_{\text{returned}}=C_{\text{all feasible}}.
$$

更好的实现是两个 retrieval channel：

$$
\boxed{
C_k^{candidate}
=
C_k^{semantic}
\cup
C_k^{exact-filter}
}
$$

而 optimality certificate 只依赖可证明 exhaustive 的 channel。

最理想的是直接给 catalog 增加：

```python
filter_by_name_tokens(...)
filter_by_demographic(...)
filter_by_season(...)
filter_by_range("rating.distribution.4_star", ...)
```

这样：

$$
\text{NL}
\to
\text{structured query}
\to
\text{database predicate pushdown}
$$

比 semantic top-k 更适合你这篇 solver-grounded 论文。

---

### 2. Profile logic 仍然可能造成 false negative

现在 size profile 的处理逻辑是：

```python
group = _product_size_group(product)
expected_size = sizes.get(group) if group else None
...
if group is None or expected_size is None or actual_size is None:
    UNKNOWN
```



问题是 `_resolve_profile_constraints()` 会把用户的 `standard_sizes` 挂到所有 slot。

因此如果用户 profile 有服装 size，但当前 slot 是：

```text
headphones
coffee machine
backpack
```

`_product_size_group()` 返回 `None`，理论上这个 size profile 应该是**不适用**：

$$
N/A
$$

而不是：

$$
UNKNOWN.
$$

这两个状态非常不同。

建议四值逻辑：

$$
\boxed{
SAT,\ UNSAT,\ UNKNOWN,\ NOT\_APPLICABLE
}
$$

至少在 profile/default 层使用。

更简单地：

```python
if group is None:
    # size profile irrelevant
    pass
```

而不是 UNKNOWN。

---

### 3. `name` 的 verifier 现在太宽松

当前 `field == "name"` 时，实际验证的字符串不是 product name，而是：

```text
brand + color + suitable_season + target_demographic + name
```



这实际上把 parser 的错误转嫁给 verifier。

例如 parser 错误地产生：

```json
{"field":"name","value":"women red jacket"}
```

当前 verifier 可能因为：

```text
target_demographic = women
color = red
name = jacket
```

而判 SAT。

短期它可能提高分数，但从方法上不够干净。

更好的 IR 是：

```json
{
  "product_type": "jacket",
  "color": "red",
  "target_demographic": "women"
}
```

然后每个 predicate 独立验证。

即：

$$
\boxed{
\text{不要让 verifier 修复 parser 的 semantic decomposition}
}
$$

否则你很难解释 error source。

---

### 4. 当前 Gurobi runtime 其实不是纯 Gurobi runtime

这是我认为必须马上修的一个实验问题。

`run()` 在 `_solve_level2/_solve_level3()` 外面计时：

```python
solver_started = time.time()
...
self.metrics["solver_runtime"] = ...
```



但 Level 2 solver 结束后默认：

```python
SHOPPING_GUROBI_CROSSCHECK = '1'
```

所以又跑：

```python
super()._solve_level2(...)
```

也就是 DP oracle。

Level 3 也是默认 `'1'`，还会再跑 enumeration oracle。

所以当前：

$$
\boxed{
solver\_runtime
=
T_{\text{Gurobi}}
+
T_{\text{DP oracle}}
}
$$

这会严重污染你后面想写的：

> solver-informed agent is faster / more efficient.

开发阶段 crosscheck 很好，但 benchmark production 应该：

```text
SHOPPING_GUROBI_CROSSCHECK=0
```

默认关闭。

另外单独统计：

```python
gurobi_runtime
oracle_runtime
```

validation suite 再开启 oracle。

---

### 5. DP solver 不应该继续作为 production class 的父类实现

现在结构上：

```text
StructuredPlanningMixin
    ├── DP Level2
    ├── enumeration Level3
    ↓
StructuredShoppingFnAgent
    ↓
ShoppingFnAgent
        ├── override Level2 with Gurobi
        └── override Level3 with Gurobi
```

这导致一个文件里同时存在：

$$
\text{production solver}
+
\text{reference oracle}.
$$

例如 parent `_solve_level2()` 是 DP，parent `_solve_level3()` 是 enumeration，而 subclass 又覆盖两者。 

这是明显的结构冗余。

建议：

```text
solvers/
    gurobi.py
    reference_oracle.py
```

然后：

```python
solution = gurobi_solver.solve(problem)

if validate:
    oracle = reference_solver.solve(problem)
    assert_equivalent(solution, oracle)
```

不要靠：

```python
super()._solve_level3(...)
```

来表达 oracle。

从论文 artifact 角度，这会清楚很多。

---

### 6. Level-3 目前其实还是“枚举 coupon + Gurobi”，不是纯 compact MILP

现在 `_benchmark_coupon_frontier()` 首先调用 `_coupon_plans()`。

而 `_coupon_plans()` 的复杂度大约是：

$$
\prod_c(Q_c+1).
$$

它把所有 coupon quantity combinations 先枚举出来。

随后 Gurobi只是：

$$
\text{select one precomputed coupon plan}.
$$



所以论文如果写：

> We formulate product and coupon selection as a MILP.

严格 auditor 会说：**不完全准确**。

更准确是：

> Product selection is optimized by MILP, jointly with a precomputed coupon-policy frontier.

这本身没问题，但 scalability 不漂亮。

更好的办法不是回到 naive coupon variables，而是直接在构造 frontier 时做 DP：

$$
DP[t]
=
\text{best coupon configuration with threshold usage }t.
$$

每加入一种 coupon：

$$
DP_{new}[t+qT_c]
=
\max
\left(
DP_{old}[t]+\{q\cdot D_c\}
\right).
$$

于是只保留：

$$
\boxed{\text{one Pareto-best state per threshold}}
$$

而不是先生成整个 Cartesian product 再压缩。

你现在 `_benchmark_coupon_frontier()` 已经在“压缩”，只是压缩得太晚。

---

### 7. `product_id` tie-breaking 是一个需要警惕的 benchmark hack

现在：

```python
_add_product_tiebreaks()
```

按：

$$
product\_id_{\text{lexicographic}}
$$

做 secondary objective，以匹配 DP oracle。

工程上它解决 deterministic equivalence。

但论文层面我要问：

> 为什么用户偏好 product ID 小的商品？

答案显然是：没有。

如果官方 GT 恰好也是这个排序，则它是 evaluator tie convention；否则这是无业务意义的 optimization objective。

我更推荐：

$$
\boxed{
\text{documented catalog rank / official sort rank}
}
$$

作为 tie-break。

如果官方 evaluator 对多个同价 optimum 只接受一个 reference product，则最好额外报告：

```text
Reference Exact Match
Objective-Equivalent Match
```

这能揭示 benchmark 本身的 tie issue。

---

### 8. `infeasible → LLM reparse` 有科学方法上的风险

现在 exhaustive retrieval 得到 `INFEASIBLE_CERTIFIED` 后，你不是直接接受它，而是把：

> exhaustive catalog verification found no feasible product

反馈给 LLM，让模型重新修改 ProblemIR。

然后 `run()` 重新 retrieval。

这可能是有效 agent behavior，但从实验设计看，它已经不是单纯：

$$
NL\to IR\to Solver.
$$

而是：

$$
NL
\to IR
\to Catalog
\to Failure
\to LLM\ reinterprets\ NL.
$$

Reviewer 可能质疑：

> Are you modifying semantics until the benchmark becomes feasible?

尤其 DeepPlanning case 大概率设计为可行时，这个 signal 很强。

我建议把它明确命名成：

$$
\boxed{\text{Verifier-Guided Semantic Repair}}
$$

并做 ablation：

```text
Structured + Solver
Structured + Solver + Semantic Repair
```

而不是默认把它隐藏在主体。

---

### 9. 你已经修好了 LLM token accounting，但还需要区分 logical calls 和 API attempts

现在 `_call_llm()` 会准确累计成功 response 的：

```text
llm_calls
input_tokens
output_tokens
total_tokens
```

这是很好的。

但底层 `call_llm()` 有 retry：

$$
1.5,\ 3,\ 6,\dots
$$

最多 5 次。

所以：

```text
llm_calls = 1
```

可能实际上发生过：

```text
API attempts = 3
```

如果论文 claim 是：

> fewer LLM calls

你要说明是：

$$
\text{successful logical model calls}
$$

还是：

$$
\text{actual API requests}.
$$

建议多一个：

```python
api_attempts
```

避免 reviewer 挑这一点。

---

### 10. 还有几个可以直接删的工程冗余

`system_prompt` 在 structured main path 基本不参与 semantic parsing；parser使用独立的 `PARSER_PROMPT`，fallback又使用固定 `FALLBACK_REACT_PROMPT_LEVEL_*`。因此它主要是 compatibility/interface baggage。

另外初始化中：

```python
self.tools_schema = self._load_tool_schemas()
...
if not Path(self.tool_schema_path).exists():
```

文件存在性检查发生在打开文件**之后**，因此这个检查基本没有意义。

还有运行时不断：

```python
sys.path.insert(...)
import tools
import base_shopping_tool
```

并且 batch runner 是多线程。这种 process-global import-path mutation 不适合作为最终 artifact。

---

## 我认为现在最值得做的版本不是 `shopping_agent(6).py` 再加 500 行

应该开始**减法重构**。

我会把最终架构收敛为：

$$
\boxed{
Parser
\rightarrow
QueryCompiler
\rightarrow
CandidateRetriever
\leftrightarrow
Optimizer
\rightarrow
Executor
\rightarrow
Verifier
}
$$

其中最重要的是新增一个真正的 `QueryCompiler`：

```python
slot constraints
    ↓
push-down predicates
    ↓
catalog candidate IDs
    ↓
remaining expensive predicates
    ↓
verified candidate set
```

这样你现在的：

```text
brand/color/size/range filter
+ Python verifier
```

会升级成真正类似 database optimizer 的：

$$
\boxed{
\text{predicate pushdown + late materialization}
}
$$

这比“多调用几次 search”更干净，也更有研究价值。

对于 L2/L3，再进一步：

$$
\boxed{
Retrieve_K
\rightarrow Solver
\rightarrow Bound/Conflict
\rightarrow Expand
\rightarrow Solver
}
$$

而不是现在：

$$
\boxed{
Enumerate\ all
\rightarrow Gurobi.
}
$$

---

### 如果这周只能改四处

我会按这个顺序做：

$$
\boxed{
1.\ \text{补齐 deterministic name/rating/season/demographic filters}
}
$$

这是最可能直接提升 **Total matched products** 的。

$$
\boxed{
2.\ \text{关闭 production DP cross-check}
}
$$

否则效率实验不干净。

$$
\boxed{
3.\ \text{修 profile applicability + size validation}
}
$$

尤其不要让非服装商品因为 profile size 进入 UNKNOWN。

$$
\boxed{
4.\ \text{把 exhaustive L2/L3 retrieval 改为 progressive candidate expansion}
}
$$

这是最可能把这套工作从“一个很强的 DeepPlanning engineering solution”提升成真正有方法贡献的：

$$
\boxed{\text{solver-informed agentic planning framework}.}
$$

现在 Gurobi 已经不是我最担心的部分；**candidate construction 才是当前方法的瓶颈和下一步创新点**。
