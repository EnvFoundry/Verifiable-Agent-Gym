import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { GymClient, GymRemoteError, PythonCLI } from "../dist/index.js";
import { collectRC } from "../examples/collect-rc.mjs";

const options = { python: process.env.PYTHON ?? "python" };
test("Persistent Python service keeps request IDs, tool feedback and terminal results", async () => {
  const root = await mkdtemp(join(tmpdir(), "gym sdk "));
  const client = new GymClient({ ...options, output: root });
  try {
    assert.ok((await client.tasks()).families.includes("rc-identification"));
    await assert.rejects(client.open({ family: "not-a-task" }), GymRemoteError);
    const [first, second] = await Promise.all([
      client.open({ family: "rc-identification", seed: 3, difficulty: 1 }),
      client.open({ family: "rc-identification", seed: 4, difficulty: 1 }),
    ]);
    assert.notEqual(first.episode_id, second.episode_id);
    const [response, cancelled] = await Promise.all([
      client.action(first.episode_id, { tool: "measure_response", arguments: { frequency_hz: 1000 } }),
      client.closeEpisode(second.episode_id),
    ]);
    assert.equal(response.type, "tool_result");
    assert.equal(cancelled.status, "cancelled");
    const cutoff = -1000 * response.result.real / response.result.imag;
    const gain = response.result.real * (1 + (1000 / cutoff) ** 2);
    const terminal = await client.action(first.episode_id, { submit: { gain, cutoff_hz: cutoff } });
    assert.equal(terminal.type, "terminal");
    assert.equal(terminal.result.status, "passed");
    assert.equal(terminal.result.reward, 1);
  } finally { await client.close(); await rm(root, { recursive: true, force: true }); }
});

test("Node batched policy collects once per attempt across Python continuation", async () => {
  const root = await mkdtemp(join(tmpdir(), "gym collection sdk "));
  try {
    const { first, resumed } = await collectRC(join(root, "study"), options);
    assert.equal(first.completed, 2);
    assert.equal(first.remaining, 2);
    assert.equal(resumed.completed, 4);
    assert.equal(resumed.counts.passed, 4);
    assert.equal(resumed.skipped_completed, 2);
    assert.equal(resumed.started_this_session, 2);
  } finally { await rm(root, { recursive: true, force: true }); }
});

test("A timed out Python command terminates instead of keeping the harness alive", async () => {
  const cli = new PythonCLI("timeit", { ...options, timeoutMs: 100 });
  await assert.rejects(cli.text(["-n", "1", "-r", "1", "import time; time.sleep(30)"]), /exceeded/);
});

test("AbortSignal and response bounds are enforced", async () => {
  const controller = new AbortController();
  const cli = new PythonCLI("timeit", options);
  const pending = cli.text(["-n", "1", "-r", "1", "import time; time.sleep(30)"], { signal: controller.signal });
  controller.abort();
  await assert.rejects(pending, /aborted/);
  const small = new GymClient({ ...options, output: await mkdtemp(join(tmpdir(), "gym bound sdk ")) });
  try {
    await assert.rejects(small.tasks({ maxOutputBytes: 8 }), /exceeds/);
    assert.ok((await small.tasks()).families.length > 0);
  } finally { await small.close(); await rm(small.options.output, { recursive: true, force: true }); }
});
