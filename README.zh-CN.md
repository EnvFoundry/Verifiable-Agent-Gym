# Verifiable Agent Gym

**可复用的任务分布、可执行环境，以及供智能体学习的交互反馈。**

[English](README.md) · [经验收集](docs/collection.md) · [训练接口](docs/training.md) · [任务包](docs/task-packages.md) · [安装](docs/install.md)

智能体观察任务、使用工具改变环境并提交结果；Gym 执行交互，并独立于策略评价结果。模型评价、合成经验、后训练与科学实验共用同一套 Episode 实现。

## 收集经验，并在中断后继续

需要 Python 3.11 或更新版本。Python 核心只依赖标准库。

```sh
git clone https://github.com/EnvFoundry/Verifiable-Agent-Gym.git
cd Verifiable-Agent-Gym
python -m pip install -e .
python examples/collect_policy.py --output runs/collection-example
```

示例为 RC 参数识别建立有限任务集，每项任务尝试四次；通过持续运行的策略进程驱动并发环境，完成部分任务后停止，再恢复并导出结果。示例使用编写好的参考求解器，将策略替换为模型推理即可收集模型交互。

```sh
agent-gym suite --tasks rc-identification --count 4 --attempts 4 --name rc-policy-study --output suite.json
agent-gym collect suite.json --reference --policy-name scripted-reference --policy-revision 1 --slots 4 --limit 5 --output runs/study
agent-gym collect suite.json --reference --policy-name scripted-reference --policy-revision 1 --slots 4 --resume --output runs/study
agent-gym collection runs/study
agent-gym rollouts runs/study --status passed --limit 10
agent-gym export-rollouts runs/study --output runs/study.jsonl
```

恢复时沿用已经完成的任务尝试，中断的交互从新环境重新开始。恢复的是实验进度；模拟器内部的活动状态由具体任务实现。[经验收集说明](docs/collection.md)定义策略协议、调度、运行会话和导出行为。

## 由自己的学习器驱动

| 接口 | 职责 |
| --- | --- |
| `TaskSuite` / `SuiteSampler` | 有限任务定义、成组尝试、确定性顺序、rank 分配和发行游标。 |
| `RolloutBatch` | 并发环境槽位，分别 reset、step、close，完成后可独立补充任务。 |
| Collection | 驱动批量策略、持续补充空闲槽位、保存进度与恢复实验。 |
| JSONL 策略进程 | 跨语言交换成批的观测和行动。 |
| 轨迹导出 | 保留任务、尝试、策略、交互、策略统计和终止反馈。 |

```sh
agent-gym collect suite.json --command-json '["/absolute/path/to/python", "/absolute/path/to/policy.py"]' --policy-name my-policy --policy-revision checkpoint-1000 --slots 8 --output runs/model
```

学习器负责推理、模型概率和优化；Gym 负责环境变化与结果。策略提供的统计保留在对应步骤中，相关尝试保留任务组关系。[训练接口](docs/training.md)提供 Python 控制方式，[交互协议](docs/protocol.md)解释单次 Episode 与终止状态。

[TypeScript SDK](sdk/typescript/README.md)提供类型化 CLI 方法、持续运行的 JSONL 环境客户端和批量策略辅助函数，可从 Node Harness 直接驱动同一套实现。

## 任务环境

| 任务族 | 智能体完成的工作 | 结果评价 |
| --- | --- | --- |
| `evidence-aggregation` | 选择有效文档版本、换算单位并组合测量。 | 来源选择、逆方差估计和标准误差。 |
| `rc-identification` | 选择测量频率，观察带噪仪器，识别参数。 | 参数误差与未参与拟合的频率响应。 |
| `thin-film-design` | 选择材料与厚度，设计减反射膜。 | 在目标波段独立计算光学响应。 |
| 数据集 SQL 任务 | 查看数据库结构，查询并提交 SQL 或结果行。 | 按规定的排序与数值容差比较结果。 |

生成任务提供可重复实例、三个难度和分离的划分命名空间；数据集任务使用明确记录。任务提供者定义观测、行动和可接受结果。参考策略展示一种解法，环境也接受其他符合目标的解法。

```sh
agent-gym --task-pack warehouse-sql=examples/sql-workload/tasks.jsonl list
python examples/sql-workload/agent.py --output runs/sql-example
```

[任务包说明](docs/task-packages.md)介绍扩展接口和独立标定任务示例。外部数据产品、模拟器与学习器通过文件或适配器接入，默认研究流程可以独立运行。

## Agent 接入

```sh
python -m pip install -e '.[mcp]'
agent-gym connect --output agent-gym.mcp.json
```

MCP 支持任务发现、交互环境、任务集、经验收集和有界轨迹分页。方法资源与提示词引导任务构建和反馈解释，仓库同时提供插件配置与共用 Skill。[安装说明](docs/install.md)覆盖 MCP Host、Codex、Claude Code 与 TypeScript 客户端。

目前使用本地线程执行环境槽位，子进程执行命令策略；远程模拟器和模型服务可以接在任务或策略接口后。代码位于 `src/agent_gym/` 与 `sdk/typescript/`，示例均可执行。贡献前阅读 [AGENTS.md](AGENTS.md)。MIT 许可证。
