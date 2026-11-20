# Verifiable Agent Gym for TypeScript harnesses

`GymClient` drives a persistent Python environment service. `GymCLI` starts finite suite/collection jobs. `serveBatchPolicy` lets a Node model or harness produce actions for Python's batched collector. The package has no runtime npm dependencies and can be used independently.

From the repository root:

```sh
python -m pip install -e .
cd sdk/typescript
npm ci
npm run build
node examples/collect-rc.mjs
```

The example runs a Node analytic RC policy through the actual Python environments. It collects two of four attempts, resumes the other two, and retains four passing outcomes. It demonstrates the interface with a noiseless instrument; it is not an LLM benchmark result.

```ts
import { GymClient, type Action } from "@envfoundry/verifiable-agent-gym";

const gym = new GymClient({ python: "/path/to/venv/bin/python", output: "/research/episodes" });
try {
  const task = await gym.open({ family: "rc-identification", seed: 7, difficulty: 1 });
  const feedback = await gym.action(task.episode_id, {
    tool: "measure_response", arguments: { frequency_hz: 1000 },
  });
  // Pass the observation and available tool schemas to your harness here.
  await gym.closeEpisode(task.episode_id);
} finally {
  await gym.close();
}
```

`tasks`, `open`, `action` and `closeEpisode` preserve the Python [episode protocol](../../docs/protocol.md). The service processes JSONL requests serially. Responses are matched to request IDs even when the host has multiple outstanding calls. Across-slot parallel collection is handled by `GymCLI.collect`, using Python `RolloutBatch`.

A batched policy is a normal Node program:

```ts
import { serveBatchPolicy, type BatchAction } from "@envfoundry/verifiable-agent-gym";

await serveBatchPolicy(async ({ observations, completed }) => {
  // Use rollout_id for conversation state; slots can be reused.
  // Release completed contexts, and batch model inference over observations.
  const actions: BatchAction[] = await myHarness(observations, completed);
  return actions;
});
```

Return one `{ slot, action, policy_data? }` for every observed slot. The final request can contain only `completed`, with no observations; return `[]`. `sample_seed` is a decimal string to preserve all 63 bits; use `BigInt` if the policy needs it. Write policy logs to stderr; stdout contains only protocol replies. See [collection](../../docs/collection.md) and the complete [RC policy](examples/rc-policy.mjs).

`GymCLI.suite` writes a finite task suite. `GymCLI.collect` accepts its path, output directory, policy identity and command array, plus slot/rank/budget/resume options. A continuation preserves completed attempts and starts fresh environments for unfinished work. Recorded collection errors throw `ProcessError` with the JSON report retained in `stdout`.

Clients accept `cwd`, environment overrides and an explicit Python executable. Per-call `{ signal, timeoutMs, maxOutputBytes }` bounds waiting and response size. Cancelling or timing out an interactive request closes its whole service, because an in-flight action cannot be undone. Call `close()` in `finally` so Python finishes active episodes and flushes their records. Normal remote operation errors are `GymRemoteError` and leave the service usable.

`npm test` exercises interactive episodes, Node-driven continuation, cancellation and bounded responses against Python. `npm pack` produces ESM JavaScript and declaration files for installation in a Node harness. Node 20 or newer is required. The package is supplied in this repository; Node process behavior is documented [here](https://nodejs.org/api/child_process.html).
