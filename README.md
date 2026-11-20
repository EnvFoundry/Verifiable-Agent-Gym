# Verifiable Agent Gym

**Reusable task distributions, executable environments and observed feedback for learning agents.**

[中文](README.zh-CN.md) · [Collection](docs/collection.md) · [Training interface](docs/training.md) · [Task packages](docs/task-packages.md) · [Installation](docs/install.md)

An agent observes a task, acts through environment tools and submits a result. Gym executes those interactions and evaluates the outcome independently of the policy. It serves model evaluation, synthetic experience collection, post-training and scientific experiments through one episode implementation.

## Collect experience and continue after interruption

Python 3.11 or newer. The Python core uses the standard library.

```sh
git clone https://github.com/EnvFoundry/Verifiable-Agent-Gym.git
cd Verifiable-Agent-Gym
python -m pip install -e .
python examples/collect_policy.py --output runs/collection-example
```

The example creates a finite RC-identification suite with four attempts per task, drives concurrent slots from a persistent policy process, stops after a partial collection, resumes and exports the results. The included policy is a scripted reference solver; replace it with model inference to collect model experience.

```sh
agent-gym suite --tasks rc-identification --count 4 --attempts 4 --name rc-policy-study --output suite.json
agent-gym collect suite.json --reference --policy-name scripted-reference --policy-revision 1 --slots 4 --limit 5 --output runs/study
agent-gym collect suite.json --reference --policy-name scripted-reference --policy-revision 1 --slots 4 --resume --output runs/study
agent-gym collection runs/study
agent-gym rollouts runs/study --status passed --limit 10
agent-gym export-rollouts runs/study --output runs/study.jsonl
```

Completed tickets are retained when collection resumes. Interrupted interactions start in a fresh environment; continuation does not claim restoration of hidden simulator state. A finite issuance limit can leave resumable work for the next invocation. See [collection semantics](docs/collection.md) for the persistent policy protocol, scheduling, recorded sessions and export behavior.

## Drive the environment from your own learner

| Interface | Role |
| --- | --- |
| `TaskSuite` / `SuiteSampler` | Finite task definitions, grouped attempts, deterministic ordering, rank assignment and a restorable issuance cursor. |
| `RolloutBatch` | Concurrent slots with selective reset, step and close; completed slots can be refilled independently. |
| Collection | Drives a batch policy, keeps slots occupied, records progress and resumes a named experiment. |
| JSONL policy process | Exchanges batches of observations and actions with a model or harness in another language. |
| Rollout export | Preserves task, attempt, policy, interactions, supplied policy statistics and terminal feedback. |

```sh
agent-gym collect suite.json --command-json '["/absolute/path/to/python", "/absolute/path/to/policy.py"]' --policy-name my-policy --policy-revision checkpoint-1000 --slots 8 --output runs/model
```

The learner owns model inference, token probabilities and optimization. Gym owns the environment transition and outcome. Policy-supplied statistics remain attached to their steps. Related attempts stay in their task group. [Training integration](docs/training.md) shows direct Python control; [the protocol](docs/protocol.md) defines the single-episode exchange and terminal statuses.

The [TypeScript SDK](sdk/typescript/README.md) includes typed CLI methods, a persistent JSONL environment client and a batch-policy helper. It invokes the same Python implementation and can be used from a Node harness without an HTTP service.

## Task environments

| Family | Agent's work | Outcome measurement |
| --- | --- | --- |
| `evidence-aggregation` | Select eligible document revisions, convert units and combine measurements. | Evidence membership, inverse-variance estimate and standard error. |
| `rc-identification` | Select frequencies, observe a noisy instrument and identify parameters. | Parameter error and response at held-out frequencies. |
| `thin-film-design` | Choose materials and layer thicknesses for an antireflection coating. | Independent characteristic-matrix calculation over the target band. |
| Dataset-backed SQL | Inspect a database, query it and submit SQL or rows. | Result equivalence with declared ordering and numerical tolerance. |

Generated families provide repeatable instances, three difficulties and separate split namespaces. Dataset-backed tasks enumerate explicit records. A task provider defines the observations, available actions and acceptable outcomes; a reference policy demonstrates one solution without restricting other valid strategies.

```sh
agent-gym --task-pack warehouse-sql=examples/sql-workload/tasks.jsonl list
python examples/sql-workload/agent.py --output runs/sql-example
```

[Task packages](docs/task-packages.md) explains the provider interface and the independent calibration-package example. Data releases and trajectory consumers can connect through files or adapters; the default research workflows run without another EnvFoundry project.

## Agent access

```sh
python -m pip install -e '.[mcp]'
agent-gym connect --output agent-gym.mcp.json
```

MCP supports task discovery, interactive episodes, suites, collection and bounded rollout pages. Method resources and prompts help design task populations and interpret feedback. Native plugin manifests and a shared operating Skill are included. [Installation](docs/install.md) covers MCP hosts, Codex, Claude Code and the TypeScript client.

Execution is local: environment slots run in threads and command policies run in subprocesses. Remote simulators and model serving can stay behind task/policy adapters. The collector records recoverable experiment progress; restoring arbitrary live environments requires a provider-specific implementation.

Implementation lives in `src/agent_gym/` and `sdk/typescript/`; examples are executable and authored. Read [AGENTS.md](AGENTS.md) before contributing. MIT license.
