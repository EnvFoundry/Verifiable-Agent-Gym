# Task distributions and batched rollouts

An external trainer controls model inference and optimization. The Gym supplies task samples, independent environment slots, observations and outcome feedback. `TaskSuite` and `RolloutBatch` make these operations explicit.

For a complete persistent batch-policy loop, use [collection](collection.md). It schedules a finite suite, refills slots, records progress and resumes completed tickets. The direct interfaces below remain useful when the learner already owns its scheduler.

## Define the task population

```python
from agent_gym.taskpacks import load_taskpack
from agent_gym.suite import TaskSuite

pack = load_taskpack("examples/sql-workload/tasks.jsonl", family="warehouse-sql")
suite = TaskSuite(pack.specs("train"), attempts_per_task=4,
                  metadata={"taskpack": pack.describe()})
sampler = suite.sampler(seed=17, epoch=0, rank=0, world_size=1)
tickets = sampler.take(4)
```

A suite binds task specifications, attempts per task and metadata to a content-derived identity. Put dataset and environment versions in the metadata. The sampler partitions whole task groups across ranks so multiple attempts for one task remain together. Each ticket carries task, group, rollout and sample-seed identities. Pass its sample seed to a policy adapter when supported.

`state_dict()` / `load_state_dict()` restore task issuance for the same suite and sampler settings. A trainer checkpoints its in-flight tickets alongside this state. Restoring the sampler does not restore a live simulator or database connection.

## Drive independent slots

```python
from agent_gym.rollout import PolicyVersion, RolloutBatch

with RolloutBatch("runs/learner", slots=4,
                  policy=PolicyVersion("policy-name", "checkpoint-1200", {"temperature": 0.7}),
                  task_factory=pack.make) as batch:
    responses = batch.reset(dict(enumerate(tickets)))
    observations = {slot: response["observation"] for slot, response in responses.items()}
    actions = policy_generate(observations)
    responses = batch.step(actions)
    # Continue active slots, consume terminal outcomes, and refill completed slots.
```

Mappings permit partial batches. Different slots execute in a thread pool; each episode's actions remain ordered. A completed slot can be refilled while other episodes continue. In-process callbacks own their operation deadlines; task packages can call remote workers or subprocess adapters when needed.

| Response field | Meaning |
| --- | --- |
| `observation` | The environment result or terminal response. |
| `done` | This attempt has ended. |
| `terminated` | The verifier returned a scored submission. |
| `truncated` | The episode's action/time budget ended. |
| `scored`, `reward` | Whether a learning score exists; unscored outcomes retain null. |
| `status` | running, passed, failed, truncated, error or cancelled. |
| `rollout` | Suite/task/attempt identity, policy revision/configuration and optional dataset reference. |

The final group ID binds task group, actual dataset reference, policy configuration and episode time/action limits. The rollout records those limits explicitly. This preserves the conditions needed for within-group comparisons. Supply model statistics as `batch.step(actions, policy_data={slot: {...}})`; the Gym records supplied values against the action step. The trainer defines log probabilities, masks, advantages and optimization semantics.

## Export experience

```sh
agent-gym export-rollouts runs/learner --output runs/learner.jsonl
```

Exports use the existing episode export-v1 shape with additional `rollout` and `policy_steps` fields. Every actual episode retains its ordered trace and outcome. Attempts failing before an episode exists remain in the run index/summary. `--scored-only` selects scored episodes.

The [Data Engineering](https://github.com/EnvFoundry/Data-Engineering) `gym-jsonl` connector preserves the complete exported payload. Its [experience cycle](https://github.com/EnvFoundry/Data-Engineering/blob/main/examples/experience_cycle.py) connects task release, grouped execution, trajectory import, shard composition and reader restart.

Run `python examples/batched_rollouts.py --output runs/batched` for the standalone reference-policy example. Use that path to integrate policy inference, batching and checkpointing before adding a trainer-specific exporter.
