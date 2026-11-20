import { mkdir, mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { GymCLI } from "../dist/index.js";

export async function collectRC(output, options = {}) {
  await mkdir(output, { recursive: false });
  const client = new GymCLI(options);
  const suite = join(output, "suite.json");
  await client.suite(suite, { tasks: ["rc-identification"], count: 2, attempts: 2, difficulty: 1, name: "typescript-rc" });
  const policy = { name: "analytic-rc", revision: "1", command: [process.execPath, fileURLToPath(new URL("rc-policy.mjs", import.meta.url))] };
  const first = await client.collect(suite, join(output, "collection"), policy, { slots: 2, limit: 2 });
  const resumed = await client.collect(suite, join(output, "collection"), policy, { slots: 2, resume: true });
  return { output, first, resumed };
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const parent = process.argv[2] ? null : await mkdtemp(join(tmpdir(), "agent-gym-typescript-"));
  const output = process.argv[2] ? resolve(process.argv[2]) : join(parent, "study");
  console.log(JSON.stringify(await collectRC(output, { python: process.env.PYTHON ?? "python" }), null, 2));
}
