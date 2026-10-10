---
language:
- en
- zh
license: apache-2.0
viewer: false
task_categories:
- text-generation
tags:
- planning
- llm-benchmark
- reasoning
- autonomous-agents
pretty_name: DeepPlanning
size_categories:
- 1k<n<10k
---

# DeepPlanning: Benchmarking Long-Horizon Agentic Planning with Verifiable Constraints

DeepPlanningBench is a challenging benchmark for evaluating long-horizon agentic planning capabilities of large language models (LLMs) with verifiable constraints. It features realistic multi-day travel planning and multi-product shopping tasks that require proactive information acquisition, local constrained reasoning, and global constrained optimization.

🌐 Website: https://qwenlm.github.io/Qwen-Agent/en/benchmarks/deepplanning/

📄 Paper: https://arxiv.org/abs/2601.18137

## Introduction

While agent evaluation has shifted toward long-horizon tasks, most benchmarks still emphasize local, step-level reasoning rather than the global constrained optimization (e.g., time and financial budgets) that demands genuine planning ability. DeepPlanning addresses this gap by introducing practical long-horizon agent planning scenarios that require:

- **Proactive Information Acquisition**: Actively gathering information through API calls to discover hidden environment states
- **Local Constrained Reasoning**: Satisfying step-level logic and specific requirements
- **Global Constrained Optimization**: Managing holistic boundaries like total budget caps and multi-day time feasibility

The benchmark includes two main domains:
- **Travel Planning**: Multi-day trip organization with tightly coupled time, location, and budget constraints
- **Shopping Planning**: Combinatorial optimization problems to find optimal products while maximizing discount utility

## Citation

If you find our work useful, please consider citing:

```bibtex
@article{deepplanning,
  title={DeepPlanning: Benchmarking Long-Horizon Agentic Planning with Verifiable Constraints},
  author={
    Yinger Zhang and Shutong Jiang and Renhao Li and Jianhong Tu and Yang Su and 
    Lianghao Deng and Xudong Guo and Chenxu Lv and Junyang Lin
  },
  journal={arXiv preprint arXiv:2601.18137},
  year={2026}
}
```