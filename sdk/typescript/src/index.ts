import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import type { Readable, Writable } from "node:stream";
import { PythonCLI, ProcessError, flag, positive, terminate, type CallOptions, type Json, type JsonObject, type ProcessOptions } from "./process.js";
export { PythonCLI, ProcessError } from "./process.js";
export type { CallOptions, Json, JsonObject, ProcessOptions } from "./process.js";

export interface TaskSpec { family: string; seed: number; split: "train" | "validation" | "test"; difficulty: 1 | 2 | 3; version: number }
export type Action = { tool: string; arguments: JsonObject } | { submit: Json };
export interface StartObservation {
  type: "start"; episode_id: string; task: TaskSpec; task_id: string; prompt: string;
  tools: { name: string; description: string; parameters: JsonObject }[];
  max_actions: number; remaining_actions: number;
}
export type EpisodeStatus = "passed" | "failed" | "truncated" | "error" | "cancelled";
export interface EpisodeResult {
  episode_id: string; task: TaskSpec; status: EpisodeStatus; reward: number | null;
  steps: number; elapsed_seconds: number; reason: string;
  verification: { passed: boolean; reward: number; reason: string; metrics: JsonObject } | null;
}
export type ActionResult = { type: "tool_result"; tool: string; result: JsonObject; remaining_actions: number }
  | { type: "tool_error"; error: string; remaining_actions: number }
  | { type: "terminal"; result: EpisodeResult; observation?: JsonObject; result_directory?: string };
export interface GymOptions extends ProcessOptions {
  output: string;
  maxActive?: number;
  episodeSeconds?: number;
  taskPacks?: Record<string, string>;
  maxPending?: number;
}

export class GymRemoteError extends Error {
  constructor(readonly remoteType: string, message: string) { super(message); this.name = "GymRemoteError"; }
}

type Pending = { resolve(value: unknown): void; reject(error: Error): void; dispose(): void; maxOutputBytes: number };

async function* lines(stream: Readable, limit: number): AsyncGenerator<string> {
  let pending: Buffer = Buffer.alloc(0);
  for await (const chunk of stream) {
    pending = Buffer.concat([pending, Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk)]);
    let end: number;
    while ((end = pending.indexOf(10)) >= 0) {
      if (end > limit) throw new ProcessError(`JSONL message exceeds ${limit} bytes`);
      const line = pending.subarray(0, end).toString("utf8");
      pending = pending.subarray(end + 1);
      if (line.trim()) yield line;
    }
    if (pending.length > limit) throw new ProcessError(`JSONL message exceeds ${limit} bytes`);
  }
  if (pending.length) throw new ProcessError("JSONL stream ended with an incomplete message");
}

/** One persistent Python Gym service. Request cancellation ends this session. */
export class GymClient {
  private readonly child: ChildProcessWithoutNullStreams;
  private readonly pending = new Map<number, Pending>();
  private readonly exited: Promise<void>;
  private sequence = 0;
  private ending = false;
  private failure?: Error;
  private stderr: Buffer = Buffer.alloc(0);
  private killTimer?: NodeJS.Timeout;
  private readonly timeoutMs: number;
  private readonly outputLimit: number;
  private readonly maxPending: number;

  constructor(readonly options: GymOptions) {
    this.timeoutMs = positive(options.timeoutMs ?? 60_000, "timeoutMs");
    this.outputLimit = positive(options.maxOutputBytes ?? 8 * 1024 * 1024, "maxOutputBytes");
    this.maxPending = positive(options.maxPending ?? 32, "maxPending");
    const errorLimit = positive(options.maxErrorBytes ?? 64 * 1024, "maxErrorBytes");
    const args = ["-u", "-m", "agent_gym"];
    for (const [name, path] of Object.entries(options.taskPacks ?? {})) args.push("--task-pack", `${name}=${path}`);
    args.push("serve", "--output", options.output);
    flag(args, "max-active", options.maxActive); flag(args, "episode-seconds", options.episodeSeconds);
    this.child = spawn(options.python ?? "python", args, {
      cwd: options.cwd, env: { ...process.env, ...options.env }, shell: false,
      windowsHide: true, detached: process.platform !== "win32", stdio: ["pipe", "pipe", "pipe"],
    });
    this.child.stderr.on("data", (part: Buffer) => { this.stderr = Buffer.concat([this.stderr, part]).subarray(-errorLimit); });
    this.child.on("error", (error) => this.fail(error));
    this.child.stdin.on("error", (error) => this.fail(error));
    this.exited = new Promise((resolve) => {
      this.child.on("close", (code, signal) => {
        if (this.killTimer) clearTimeout(this.killTimer);
        this.ending = true;
        const error = this.failure ?? new ProcessError(`Gym service closed (${code ?? signal})`, code, this.stderr.toString("utf8"));
        for (const request of this.pending.values()) { request.dispose(); request.reject(error); }
        this.pending.clear();
        resolve();
      });
    });
    void this.receive().catch((error: unknown) => this.fail(error instanceof Error ? error : new Error(String(error))));
  }

  private fail(error: Error): void {
    if (this.failure) return;
    this.failure = error;
    this.ending = true;
    for (const request of this.pending.values()) { request.dispose(); request.reject(error); }
    this.pending.clear();
    if (this.killTimer) clearTimeout(this.killTimer);
    terminate(this.child);
    this.killTimer = setTimeout(() => terminate(this.child, true), 1000);
    this.killTimer.unref();
  }

  private async receive(): Promise<void> {
    for await (const line of lines(this.child.stdout, this.outputLimit)) {
      const response = JSON.parse(line) as { id?: number; result?: unknown; error?: { type: string; message: string } };
      if (!response || typeof response !== "object" || typeof response.id !== "number") throw new ProcessError("Gym service returned a response without a request ID");
      const pending = this.pending.get(response.id);
      if (!pending) throw new ProcessError("Gym service returned an unknown request ID");
      this.pending.delete(response.id); pending.dispose();
      if (Buffer.byteLength(line) > pending.maxOutputBytes) pending.reject(new ProcessError(`Gym response exceeds ${pending.maxOutputBytes} bytes`));
      else if (response.error) pending.reject(new GymRemoteError(response.error.type, response.error.message));
      else if (Object.hasOwn(response, "result")) pending.resolve(response.result);
      else pending.reject(new ProcessError("Gym response contains neither result nor error"));
    }
  }

  async request<T = JsonObject>(method: "tasks" | "open" | "action" | "close", params: object = {}, call: CallOptions = {}): Promise<T> {
    if (this.ending) return Promise.reject(this.failure ?? new ProcessError("Gym client is closed"));
    if (this.pending.size >= this.maxPending) return Promise.reject(new ProcessError(`Gym client has ${this.maxPending} pending requests; await a response before issuing more`));
    call.signal?.throwIfAborted();
    const timeout = positive(call.timeoutMs ?? this.timeoutMs, "timeoutMs");
    const limit = positive(call.maxOutputBytes ?? this.outputLimit, "maxOutputBytes");
    const id = ++this.sequence;
    const message = JSON.stringify({ id, method, params }) + "\n";
    if (Buffer.byteLength(message) > this.outputLimit) return Promise.reject(new ProcessError("Gym request exceeds the configured message size"));
    return new Promise<T>((resolve, reject) => {
      const abort = () => this.fail(new ProcessError("Gym request aborted; session closed"));
      const timer = setTimeout(() => this.fail(new ProcessError(`Gym request exceeded ${timeout} ms; session closed`)), timeout);
      timer.unref();
      this.pending.set(id, {
        resolve: value => resolve(value as T), reject,
        maxOutputBytes: limit,
        dispose: () => { clearTimeout(timer); call.signal?.removeEventListener("abort", abort); },
      });
      call.signal?.addEventListener("abort", abort, { once: true });
      if (call.signal?.aborted) abort();
      else this.child.stdin.write(message, error => { if (error) this.fail(error); });
    });
  }

  tasks(call?: CallOptions) { return this.request<{ schema_version: number; families: string[]; splits: string[]; difficulties: number[] }>("tasks", {}, call); }
  open(spec: Pick<TaskSpec, "family"> & Partial<Omit<TaskSpec, "family">>, call?: CallOptions) { return this.request<StartObservation>("open", spec, call); }
  action(episodeId: string, action: Action, call?: CallOptions) { return this.request<ActionResult>("action", { episode_id: episodeId, action }, call); }
  closeEpisode(episodeId: string, call?: CallOptions) { return this.request<EpisodeResult>("close", { episode_id: episodeId }, call); }

  /** EOF lets Gym close its active episodes and flush their final records. */
  async close(): Promise<void> {
    if (!this.ending) {
      this.ending = true;
      this.child.stdin.end();
      this.killTimer = setTimeout(() => terminate(this.child, true), 3000);
      this.killTimer.unref();
    }
    await this.exited;
  }
}

export interface BatchObservation { slot: number; rollout_id: string; sample_seed: string; observation: StartObservation | ActionResult }
export interface BatchRequest {
  type: "batch"; schema_version: 1; observations: BatchObservation[];
  completed: { slot: number; rollout_id: string; status: EpisodeStatus; scored: boolean; reward: number | null; result: EpisodeResult }[];
}
export interface BatchAction { slot: number; action: Action; policy_data?: JsonObject }
export interface CollectionResult {
  schema_version: number; requested: number; completed: number; remaining: number;
  counts: Record<EpisodeStatus, number>; started_this_session: number; restarted_this_session: number;
  skipped_completed: number; policy_batches: number; interrupted: boolean; output: string; export_argv: string[];
  attempt_history: { attempts: number; incomplete: number; counts: Record<string, number>; recorded_episode_seconds: number };
}

/** Adapt an asynchronous model/harness function to the collector's JSONL policy protocol. */
export async function serveBatchPolicy(
  policy: (request: BatchRequest) => Promise<BatchAction[]> | BatchAction[],
  options: { input?: Readable; output?: Writable; maxMessageBytes?: number } = {},
): Promise<void> {
  const limit = positive(options.maxMessageBytes ?? 8 * 1024 * 1024, "maxMessageBytes");
  const output = options.output ?? process.stdout;
  for await (const line of lines(options.input ?? process.stdin, limit)) {
    const request = JSON.parse(line) as BatchRequest;
    if (request?.type !== "batch" || request.schema_version !== 1 || !Array.isArray(request.observations) || !Array.isArray(request.completed)) throw new Error("Expected batch policy request schema 1");
    const actions = await policy(request);
    const expected = new Set(request.observations.map(item => item.slot));
    if (actions.length !== expected.size || actions.some(item => !expected.delete(item.slot)) || expected.size) throw new Error("Return exactly one action for each observed slot");
    const response = JSON.stringify({ actions }) + "\n";
    if (Buffer.byteLength(response) > limit) throw new Error("Batch policy response exceeds maxMessageBytes");
    await new Promise<void>((resolve, reject) => output.write(response, error => error ? reject(error) : resolve()));
  }
}

/** Finite collection jobs use the CLI; interactive episodes use GymClient. */
export class GymCLI {
  readonly cli: PythonCLI;
  constructor(options: ProcessOptions = {}) { this.cli = new PythonCLI("agent_gym", options); }
  capabilities(call?: CallOptions) { return this.cli.json(["capabilities"], call); }
  guide(topic: string, call?: CallOptions) { return this.cli.json(["guide", topic], call); }
  collection(path: string, call?: CallOptions) { return this.cli.json(["collection", path], call); }
  rollouts(path: string, options: { offset?: number; limit?: number; status?: EpisodeStatus } = {}, call?: CallOptions) {
    const args = ["rollouts", path];
    flag(args, "offset", options.offset); flag(args, "limit", options.limit); flag(args, "status", options.status);
    return this.cli.json(args, call);
  }
  async tasks(call?: CallOptions): Promise<string[]> { return (await this.cli.text(["list"], call)).split(/\r?\n/).filter(Boolean); }
  packInfo(path: string, family?: string, call?: CallOptions) {
    const args = ["pack-info", path]; flag(args, "family", family); return this.cli.json(args, call);
  }
  suite(output: string, options: { tasks: string[]; count?: number; seed?: number; split?: "train" | "validation" | "test";
      difficulty?: 1 | 2 | 3; attempts?: number; name?: string }, call?: CallOptions) {
    const args = ["suite", "--tasks", ...options.tasks, "--output", output];
    for (const name of ["count", "seed", "split", "difficulty", "attempts", "name"] as const) flag(args, name, options[name]);
    return this.cli.json(args, call);
  }
  collect(suite: string, output: string, policy: { name: string; revision: string; generation?: JsonObject }
      & ({ command: string[] } | { reference: true }),
      options: { slots?: number; workers?: number; seed?: number; epoch?: number; rank?: number; worldSize?: number;
        episodeSeconds?: number; responseSeconds?: number; resume?: boolean; retryErrors?: boolean; limit?: number } = {}, call?: CallOptions) {
    const args = ["collect", suite, "--output", output, "--policy-name", policy.name, "--policy-revision", policy.revision];
    if ("command" in policy) args.push("--command-json", JSON.stringify(policy.command));
    else args.push("--reference");
    if (policy.generation) args.push("--generation", JSON.stringify(policy.generation));
    for (const name of ["slots", "workers", "seed", "epoch", "rank", "resume", "limit"] as const) flag(args, name, options[name]);
    flag(args, "world-size", options.worldSize); flag(args, "episode-seconds", options.episodeSeconds);
    flag(args, "response-seconds", options.responseSeconds); flag(args, "retry-errors", options.retryErrors);
    return this.cli.json<CollectionResult>(args, call);
  }
}
