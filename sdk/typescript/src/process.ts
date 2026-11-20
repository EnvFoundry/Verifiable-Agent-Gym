import { spawn, type ChildProcess } from "node:child_process";

export type Json = null | boolean | number | string | Json[] | { [key: string]: Json };
export type JsonObject = { [key: string]: Json };

export interface ProcessOptions {
  /** Executable path, not a shell command. Defaults to python. */
  python?: string;
  cwd?: string;
  /** Overrides merged with the current process environment. */
  env?: NodeJS.ProcessEnv;
  timeoutMs?: number;
  /** Maximum complete CLI response, in bytes. Defaults to 16 MiB. */
  maxOutputBytes?: number;
  /** Retained stderr tail, in bytes. Defaults to 64 KiB. */
  maxErrorBytes?: number;
}

export interface CallOptions {
  signal?: AbortSignal;
  timeoutMs?: number;
  maxOutputBytes?: number;
}

export class ProcessError extends Error {
  constructor(message: string, readonly exitCode: number | null = null, readonly stderr = "", readonly stdout = "") {
    super(message);
    this.name = "ProcessError";
  }
}

export function positive(value: number, name: string): number {
  if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`${name} must be a positive integer`);
  return value;
}

/** End the owned Python process group on POSIX or its process tree on Windows. */
export function terminate(child: ChildProcess, force = false): void {
  if (child.pid === undefined) return;
  try {
    if (process.platform === "win32") {
      if (child.exitCode !== null || child.signalCode !== null) return;
      const killer = spawn("taskkill", ["/pid", String(child.pid), "/t", "/f"], { windowsHide: true, stdio: "ignore" });
      killer.on("error", () => child.kill("SIGKILL"));
      killer.unref();
    }
    else process.kill(-child.pid, force ? "SIGKILL" : "SIGTERM");
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ESRCH") child.kill(force ? "SIGKILL" : "SIGTERM");
  }
}

export function flag(args: string[], name: string, value: string | number | boolean | undefined): void {
  if (value === undefined || value === false) return;
  args.push(`--${name}`);
  if (value !== true) args.push(String(value));
}

export class PythonCLI {
  constructor(readonly module: string, readonly options: ProcessOptions = {}) {}

  async json<T = JsonObject>(args: readonly string[], call: CallOptions = {}): Promise<T> {
    const text = await this.text(args, call);
    try { return JSON.parse(text) as T; }
    catch { throw new ProcessError("Python command did not return one JSON value", 0, text.slice(0, 2000)); }
  }

  async text(args: readonly string[], call: CallOptions = {}): Promise<string> {
    const timeoutMs = positive(call.timeoutMs ?? this.options.timeoutMs ?? 300_000, "timeoutMs");
    const outputLimit = positive(call.maxOutputBytes ?? this.options.maxOutputBytes ?? 16 * 1024 * 1024, "maxOutputBytes");
    const errorLimit = positive(this.options.maxErrorBytes ?? 64 * 1024, "maxErrorBytes");
    call.signal?.throwIfAborted();
    return new Promise((resolve, reject) => {
      const child = spawn(this.options.python ?? "python", ["-u", "-m", this.module, ...args], {
        cwd: this.options.cwd, env: { ...process.env, ...this.options.env },
        shell: false, windowsHide: true, detached: process.platform !== "win32",
        stdio: ["ignore", "pipe", "pipe"],
      });
      let total = 0;
      const output: Buffer[] = [];
      let stderr: Buffer = Buffer.alloc(0);
      let failure: Error | undefined;
      let killTimer: NodeJS.Timeout | undefined;
      const stop = (error: Error) => {
        if (failure) return;
        failure = error;
        terminate(child);
        killTimer = setTimeout(() => terminate(child, true), 1000);
        killTimer.unref();
      };
      const abort = () => stop(new ProcessError("Python command aborted"));
      const timer = setTimeout(() => stop(new ProcessError(`Python command exceeded ${timeoutMs} ms`)), timeoutMs);
      timer.unref();
      call.signal?.addEventListener("abort", abort, { once: true });
      child.stdout.on("data", (part: Buffer) => {
        total += part.length;
        if (total > outputLimit) stop(new ProcessError(`Python response exceeds ${outputLimit} bytes; request a smaller page or save analysis to a file`));
        else output.push(part);
      });
      child.stderr.on("data", (part: Buffer) => {
        stderr = Buffer.concat([stderr, part]).subarray(-errorLimit);
      });
      child.on("error", (error) => { failure = error; });
      child.on("close", (code, signal) => {
        clearTimeout(timer);
        if (killTimer) clearTimeout(killTimer);
        call.signal?.removeEventListener("abort", abort);
        if (failure) reject(failure);
        else if (code !== 0) reject(new ProcessError(`Python command exited ${code ?? signal}: ${stderr.toString("utf8").trim()}`, code, stderr.toString("utf8"), Buffer.concat(output).toString("utf8")));
        else resolve(Buffer.concat(output).toString("utf8").trim());
      });
      // The signal may have changed between the initial check and listener setup.
      if (call.signal?.aborted) abort();
    });
  }
}
