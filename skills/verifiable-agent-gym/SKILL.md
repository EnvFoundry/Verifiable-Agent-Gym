---
name: verifiable-agent-gym
description: Operate agent environments, build task packages, collect grouped rollouts with an external policy and resume interrupted collection. Use for scientific or data-backed tasks, agent training experience, policy comparison and Verifiable Agent Gym harness integration.
---

# Verifiable Agent Gym

Start from the task population, what the agent observes and changes, and the outcome being measured. Discover installed families with `gym_tasks` or `agent-gym list`. Read installed guidance with `agent-gym guide TOPIC` or `gym://guides/TOPIC`.

| Work | Guide | Interface |
| --- | --- | --- |
| Construct an environment and outcome measure | `task-design` | `TaskSpec`, task-provider entry point |
| Solve one task through its tools | `episode-workflow` | `gym_open`, `gym_action`, `gym_close` |
| Connect a learner and collect attempts | `grouped-collection` | `suite`, `collect`, `export-rollouts` |
| Compare policy behavior and resource use | `policy-comparison` | Matched suites and recorded trajectories |

For an episode, preserve its returned `episode_id`. Read the prompt, tool schemas, units, submission fields and action budget. Call `gym_action` with `{tool: name, arguments: {...}}`, or `{submit: {...}}` for a result. Keep dependent actions ordered. Report the actual terminal outcome and relevant measurements. Close an unfinished episode when abandoning it. Treat `failed` as a scored submission and `error`, `truncated` or `cancelled` as unscored outcomes.

For collection, create or load a finite suite and declare attempts per task and a concrete policy revision. Use `collect` with a batched policy command or Python callable. The collector fills slots and preserves task groups and logical attempt identities. Resume the same collection to retain completed outcomes; interrupted episodes restart in fresh environments. Change the output location for a different policy or sampling plan. Export trajectories with `export-rollouts` and retain model statistics only when measured by the policy.

Use installed guides when the host has only this Skill and the Python package. For source-level details, explicitly locate the [Gym checkout](https://github.com/EnvFoundry/Verifiable-Agent-Gym): `docs/collection.md` defines callable/process collection, `docs/training.md` covers direct `RolloutBatch` use, and `sdk/typescript/README.md` provides Node clients. These paths refer to the checkout. For data-backed tasks, use `docs/task-packages.md`, load the pack and choose from its actual split specifications.

Extend the shared `Task` and `Episode` interfaces. Evaluate submitted artifacts independently of the solver and accept valid alternative procedures. For policy research, distinguish first-attempt quality, success among multiple attempts and the quality of the selected answer. Keep retries and collection coverage in the resource comparison.

Example request: “Collect four attempts on each task with these two policy revisions, preserve the same interaction budget, resume existing work, and identify which observed action choices explain the outcome difference.” Return suite, run and export paths with the measured result and a few relevant trajectories.
