# Batched experience collection

`collect` runs a finite task suite through an external policy, keeps environment slots filled, and resumes unfinished work. Policy inference receives a batch of current observations. Environment steps use `RolloutBatch`, so the same episode behavior and outcome meanings apply in Python, CLI and MCP.

## Run a collection

```sh
agent-gym suite --tasks rc-identification --count 4 --attempts 4 \
  --name rc-policy-study --output suite.json
agent-gym collect suite.json --reference \
  --policy-name scripted-reference --policy-revision 1 \
  --slots 4 --limit 5 --output runs/collection
agent-gym collect suite.json --reference \
  --policy-name scripted-reference --policy-revision 1 \
  --slots 4 --resume --output runs/collection
agent-gym collection runs/collection
agent-gym export-rollouts runs/collection --output runs/trajectories.jsonl
```

This collects five of sixteen requested attempts, then continues the remaining eleven. `--limit` limits new starts in one invocation and lets those attempts finish. It does not change the suite. `--reference` runs the bundled scripted solver. Run `python examples/collect_policy.py --output runs/collection-example` for the same sequence using a persistent external policy process.

A suite names the task population and attempts per task. Generated families use explicit seeds and difficulties; `--task-pack FAMILY=PATH suite --tasks FAMILY` selects the finite records of an installed dataset pack. Python can construct the same manifest with `TaskSuite(...).save(path)`. Record environment and source revisions in suite metadata and decoding settings in `PolicyVersion.generation` or `--generation`.

`--rank` and `--world-size` partition whole task groups. All attempts for one task remain on the same rank. Each rank writes a separate collection directory. `--seed` and `--epoch` determine sampling order and policy sample seeds.

## Connect a batched policy

```sh
agent-gym collect suite.json \
  --command-json '["/absolute/path/to/python", "/absolute/path/to/policy.py"]' \
  --policy-name model-and-harness --policy-revision checkpoint-1200 \
  --generation '{"temperature":0.7}' \
  --slots 8 --output runs/checkpoint-1200
```

The command runs once per collection session with UTF-8 stdin/stdout. Its working directory is the session's `policy/` directory; stderr is retained there. Return one JSON object for each request and keep logs on stderr. `--response-seconds` bounds each command response; in-process Python policies manage their own inference deadlines.

One request contains the current observations and the outcomes completed since the previous request:

```json
{
  "schema_version": 1,
  "type": "batch",
  "observations": [
    {
      "slot": 0,
      "rollout_id": "logical-attempt-id",
      "sample_seed": "6171214536251713121",
      "observation": {"type": "start", "prompt": "...", "tools": []}
    }
  ],
  "completed": []
}
```

`sample_seed` is a decimal string on this wire protocol so JavaScript can preserve the full integer. The stored rollout retains its integer value. A policy chooses how to map that value to its inference backend's supported seed range.

Return exactly one action for every observed slot:

```json
{
  "actions": [
    {
      "slot": 0,
      "action": {"tool": "measure_response", "arguments": {"frequency_hz": 1000}},
      "policy_data": {"output_tokens": 24}
    }
  ]
}
```

An action is either `{"tool": name, "arguments": {...}}` or `{"submit": answer}`. `policy_data` is optional JSON supplied by the learner; its token statistics and other annotations are written against the actual action step. The Gym does not calculate missing model probabilities.

A completed entry contains `slot`, `rollout_id`, `status`, `scored`, `reward` and `result`. Maintain conversation state by `rollout_id`, release it on completion, and initialize a fresh context whenever an observation has `type: "start"`. Slot numbers are reused. Terminal notifications can arrive in the same batch as new observations in those slots. A final request can have an empty observations list; acknowledge it with `{"actions": []}` without making a model call.

Python uses the same request and response objects:

```python
from agent_gym import PolicyVersion, TaskSuite, collect

suite = TaskSuite.load("suite.json")

def policy(batch):
    # Release contexts in batch["completed"], then generate for all observations.
    return {"actions": [
        {"slot": row["slot"], "action": model_action(row)}
        for row in batch["observations"]
    ]}

result = collect(suite, "runs/model", policy,
                 policy_version=PolicyVersion("model-and-harness", "checkpoint-1200"),
                 slots=8)
```

The caller owns a Python policy's lifecycle. The collector owns a command process and closes it when its session ends. Different environment slots step concurrently; each policy request sees the batch after those steps return.

## Continuation and exported experience

`--resume` keeps completed `passed`, `failed`, `truncated` and `error` outcomes. Interrupted or cancelled attempts start again in fresh environments with the same suite, task, attempt and sample-seed identities. `--retry-errors` explicitly retries recorded errors. A scored wrong answer is a completed observation and is retained.

Python callers can pass a `threading.Event` as `stop`; collection observes it between batches and before applying returned actions. MCP cancellation uses the same operation and closes active episodes before returning. Command response deadlines bound an in-flight inference request.

Continuation requires the same suite, policy, sampling and episode settings. Slots are part of the configured policy batch; worker concurrency can change. The task factory and model adapter must implement the revisions named in those records. Live simulator or policy memory is not restored.

Each invocation writes a new session. Earlier traces remain available, including interrupted attempts. The root `rollout-index.jsonl` selects one outcome per logical rollout after collection returns. `export-rollouts` follows this selection, so resuming or retrying an error does not duplicate a training sample. The summary separates the selected population from `attempt_history`, which includes prior errors, cancellations, incomplete attempts and recorded episode time. That history is needed when estimating the work spent to collect the final experience.

If a process stops after an episode result was written but before the selection index was updated, continuation recovers the terminal file and retains it. It never needs to rerun an already recorded scored result. A missing result restarts in a fresh environment. Setup failures have no action trace; they appear in progress and outcome pages and are omitted from trajectory export.

## Work through an agent

`gym_suite` writes a suite, `gym_collect` executes it, and `gym_collection` returns compact progress. `gym_rollouts` reads at most 100 selected outcomes with an optional status filter. `gym_export_rollouts` creates the complete trajectory file for a consumer. MCP paths are relative to the configured output workspace.

```sh
agent-gym rollouts runs/collection --status failed --limit 20
agent-gym guide grouped-collection
```

The guides at `gym://guides/task-design`, `episode-workflow`, `grouped-collection` and `policy-comparison` connect these operations to research design. The installed prompts help define an executable task, plan collection and compare policy revisions on a common task population.
