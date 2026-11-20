"""Resumable task-suite collection through the shared batched episode runtime."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
import json
import math
from pathlib import Path
import threading
import uuid

from .contracts import Json, Task, TaskSpec
from .process import CommandAgent
from .registry import make
from .rollout import PolicyVersion, RolloutBatch
from .storage import RunLease, atomic_json, dumps, json_copy
from .suite import TaskSuite


STATUSES = ("passed", "failed", "truncated", "error", "cancelled")


class CommandBatchPolicy:
    """One persistent JSONL process for a collection session's inference batches."""

    def __init__(self, command: list[str], workspace: str | Path, *, timeout: float = 60):
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("policy timeout must be finite and positive")
        self.timeout = timeout
        self.transport = CommandAgent(command, Path(workspace))

    def __call__(self, request: Json) -> Json:
        return self.transport.act(request, timeout=self.timeout)

    def close(self) -> None:
        self.transport.close()


class ReferenceBatchPolicy:
    """Batched wrapper around the bundled observation-only reference policies."""

    def __init__(self):
        self.agents = {}

    def __call__(self, request: Json) -> Json:
        from .reference import ReferenceAgent
        for row in request["completed"]:
            agent = self.agents.pop(row["rollout_id"], None)
            if agent is not None:
                agent.close()
        actions = []
        for row in request["observations"]:
            if row["rollout_id"] not in self.agents or row["observation"].get("type") == "start":
                previous = self.agents.pop(row["rollout_id"], None)
                if previous is not None:
                    previous.close()
                self.agents[row["rollout_id"]] = ReferenceAgent()
            agent = self.agents[row["rollout_id"]]
            actions.append({"slot": row["slot"], "action": agent.act(row["observation"])})
        return {"actions": actions}

    def close(self) -> None:
        for agent in self.agents.values():
            agent.close()
        self.agents.clear()


def _positive(value, name, *, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _responses(value: Json, active: set[int]) -> tuple[dict, dict]:
    """Normalize one batch before performing any environment actions."""
    if not isinstance(value, dict) or set(value) != {"actions"} or not isinstance(value["actions"], list):
        raise ValueError("batch policy must return an object containing an actions list")
    actions, annotations = {}, {}
    for row in value["actions"]:
        if not isinstance(row, dict) or not {"slot", "action"} <= row.keys() or row.keys() - {"slot", "action", "policy_data"}:
            raise ValueError("each policy action requires slot and action, with optional policy_data")
        slot = row["slot"]
        if type(slot) is not int or slot not in active or slot in actions:
            raise ValueError("policy actions must name each active slot exactly once")
        if not isinstance(row["action"], dict):
            raise ValueError("a policy action must be an object")
        actions[slot] = json_copy(row["action"])
        if "policy_data" in row:
            if not isinstance(row["policy_data"], dict):
                raise ValueError("policy_data must be an object")
            annotations[slot] = json_copy(row["policy_data"])
    if set(actions) != active:
        raise ValueError("policy actions must name each active slot exactly once")
    return actions, annotations


def _snapshot(root: Path) -> tuple[dict[str, Json], set[str], Json]:
    """Recover terminal files, including results written just before a crash."""
    selected: dict[str, Json] = {}
    known = set()
    history = Counter()
    elapsed = 0.0
    for session in sorted((root / "sessions").glob("*")):
        if not session.is_dir():
            continue
        for path in sorted(session.glob("*/*/rollout.json")):
            rollout = json.loads(path.read_text(encoding="utf-8"))
            identifier = rollout["rollout_id"]
            known.add(identifier)
            history["attempts"] += 1
            result_path = path.parent / "result.json"
            if result_path.exists():
                result = json.loads(result_path.read_text(encoding="utf-8"))
                history[result["status"]] += 1
                elapsed += result.get("elapsed_seconds", 0)
                selected[identifier] = {"rollout_id": identifier, "path": path.parent.relative_to(root).as_posix(),
                                        "setup_error": False, "status": result["status"]}
            else:
                history["incomplete"] += 1
                selected.pop(identifier, None)
        for path in sorted((session / "setup-errors").glob("*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            identifier = record["rollout"]["rollout_id"]
            known.add(identifier)
            history["attempts"] += 1
            history[record["result"]["status"]] += 1
            selected[identifier] = {"rollout_id": identifier, "path": path.relative_to(root).as_posix(),
                                    "setup_error": True, "status": record["result"]["status"]}
    return selected, known, {"attempts": history["attempts"], "incomplete": history["incomplete"],
                              "counts": {status: history[status] for status in STATUSES},
                              "recorded_episode_seconds": elapsed}


def _write_selection(root: Path, selected: dict[str, Json]) -> None:
    # The journal has one selected attempt per logical rollout. Historical
    # attempts remain in sessions/ for work/cost accounting.
    pending = root / ".rollout-index.pending"
    with pending.open("w", encoding="utf-8", newline="\n") as output:
        for identifier in sorted(selected):
            row = selected[identifier]
            output.write(dumps({key: row[key] for key in ("rollout_id", "path", "setup_error")}) + "\n")
    pending.replace(root / "rollout-index.jsonl")


def _report(root: Path, selected: dict[str, Json], total: int, *, started=0, restarted=0,
            skipped=0, batches=0, interrupted=False) -> Json:
    counts = Counter(row["status"] for row in selected.values())
    completed = sum(counts[status] for status in STATUSES if status != "cancelled")
    return {"schema_version": 1, "requested": total, "completed": completed,
            "remaining": total - completed, "counts": {status: counts[status] for status in STATUSES},
            "started_this_session": started, "restarted_this_session": restarted,
            "skipped_completed": skipped, "policy_batches": batches, "interrupted": interrupted,
            "output": str(root), "export_argv": ["agent-gym", "export-rollouts", str(root), "--output", "trajectories.jsonl"]}


def collection_info(root: str | Path) -> Json:
    """Return the persisted population, execution settings and compact progress."""
    root = Path(root).resolve()
    manifest = json.loads((root / "collection.json").read_text(encoding="utf-8"))
    summary_path = root / "collection-summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None
    return {"schema_version": 1, "suite": {"name": manifest["suite"]["name"],
             "tasks": len(manifest["suite"]["tasks"]), "attempts_per_task": manifest["suite"]["attempts_per_task"]},
            "policy": manifest["policy"], "sampling": {key: value for key, value in manifest["sampling"].items()
                                                         if key not in ("suite_id", "schema_version")},
            "execution": manifest["execution"], "summary": summary}


def collection_rollouts(root: str | Path, *, offset: int = 0, limit: int = 20,
                        status: str | None = None) -> Json:
    """Page selected rollout outcomes without returning action histories."""
    _positive(offset, "offset", minimum=0)
    _positive(limit, "limit")
    if limit > 100 or status is not None and status not in STATUSES:
        raise ValueError("limit must be <= 100 and status must name a terminal outcome")
    root = Path(root).resolve()
    items, total = [], 0
    with (root / "rollout-index.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            entry = json.loads(line)
            path = root / entry["path"]
            if entry["setup_error"]:
                saved = json.loads(path.read_text(encoding="utf-8"))
                record, result = saved["rollout"], saved["result"]
            else:
                record = json.loads((path / "rollout.json").read_text(encoding="utf-8"))
                result = json.loads((path / "result.json").read_text(encoding="utf-8"))
            if status is not None and result["status"] != status:
                continue
            if offset <= total < offset + limit:
                items.append({"rollout_id": record["rollout_id"], "task": record["task"],
                              "attempt": record["attempt"], "policy": record["policy"],
                              "status": result["status"], "reward": result["reward"],
                              "steps": result.get("steps", 0), "elapsed_seconds": result.get("elapsed_seconds"),
                              "path": entry["path"]})
            total += 1
    end = offset + len(items)
    return {"items": items, "total": total, "offset": offset, "limit": limit,
            "next_offset": end if end < total else None}


def collect(suite: TaskSuite | str | Path, output: str | Path,
            policy: Callable[[Json], Json] | None = None, *, policy_version: PolicyVersion,
            command: list[str] | None = None, slots: int = 4, workers: int | None = None,
            seed: int = 0, epoch: int = 0, rank: int = 0, world_size: int = 1,
            shuffle: bool = True, episode_seconds: float = 120, response_seconds: float = 60,
            resume: bool = False, retry_errors: bool = False, limit: int | None = None,
            stop: threading.Event | None = None,
            task_factory: Callable[[TaskSpec], Task] = make) -> Json:
    """Collect a fixed rank of a suite, refill slots, and continue prior work.

    The policy receives one JSON batch and returns one action per active slot.
    A callable is caller-owned; a command process is owned by this operation.
    limit caps newly started logical rollouts in this invocation and drains
    them before returning. Resume reuses completed outcomes and restarts
    cancelled/incomplete attempts in fresh environments with the same ticket.
    """
    if not isinstance(suite, TaskSuite):
        suite = TaskSuite.load(suite)
    if (policy is None) == (command is None):
        raise ValueError("provide exactly one batch policy callable or command")
    if policy is not None and not callable(policy):
        raise ValueError("policy must be callable")
    if command is not None and (not isinstance(command, list) or not command or
                                any(not isinstance(arg, str) or not arg for arg in command)):
        raise ValueError("command must be a nonempty argv list")
    _positive(slots, "slots")
    workers = slots if workers is None else workers
    _positive(workers, "workers")
    if limit is not None:
        _positive(limit, "limit")
    for value, name in ((episode_seconds, "episode_seconds"), (response_seconds, "response_seconds")):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if any(type(value) is not bool for value in (resume, retry_errors)):
        raise ValueError("resume and retry_errors must be booleans")
    if retry_errors and not resume:
        raise ValueError("retry_errors requires resume")
    sampler = suite.sampler(seed=seed, epoch=epoch, rank=rank, world_size=world_size, shuffle=shuffle)
    sampling = sampler.state_dict()
    sampling.pop("cursor")
    # Package releases and scheduling concurrency do not identify policy or
    # environment semantics; users pin those in the suite and policy records.
    manifest = {"schema_version": 1, "suite": suite.to_dict(), "policy": policy_version.to_dict(),
                "sampling": sampling, "execution": {"slots": slots, "episode_seconds": episode_seconds,
                "response_seconds": response_seconds}, "command": command}
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with RunLease(root):
        manifest_path = root / "collection.json"
        if manifest_path.exists():
            if not resume:
                raise ValueError("output already contains a collection; use resume or a new directory")
            if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
                raise ValueError("resume requires the same suite, policy, sampling and execution settings")
        else:
            if resume:
                raise ValueError("no collection exists to resume")
            if (root / "rollout-run.json").exists() or (root / "run.json").exists() or (root / "sessions").exists():
                raise ValueError("output contains a different run")
            atomic_json(manifest_path, manifest)
        selected, known, _ = _snapshot(root)
        completed = {key for key, row in selected.items()
                     if row["status"] != "cancelled" and not (retry_errors and row["status"] == "error")}
        eligible = (ticket for ticket in sampler if ticket.rollout_id not in completed)
        started, restarted, batches = 0, 0, 0
        interrupted = False
        exhausted = False
        previous_sessions = sorted((root / "sessions").glob("*"))
        sequence = 1 + max((int(path.name.split("-", 1)[0]) for path in previous_sessions), default=0)
        session = root / "sessions" / f"{sequence:08d}-{uuid.uuid4().hex[:8]}"
        owned_policy = None
        try:
            with RolloutBatch(session, slots=slots, workers=workers, policy=policy_version,
                              episode_seconds=episode_seconds, task_factory=task_factory) as batch:
                observations: dict[int, Json] = {}
                finished = []

                def update(rows):
                    for slot, row in rows.items():
                        if row["done"]:
                            record = row["rollout"]
                            selected[record["rollout_id"]] = {
                                "rollout_id": record["rollout_id"],
                                "path": (session / record["path"]).relative_to(root).as_posix(),
                                "setup_error": record.get("episode_id") is None, "status": row["status"]}
                            observations.pop(slot, None)
                            finished.append({"slot": slot, "rollout_id": row["rollout"]["rollout_id"],
                                             "status": row["status"], "scored": row["scored"],
                                             "reward": row["reward"], "result": row["result"]})
                        else:
                            observations[slot] = row

                try:
                    while True:
                        if stop is not None and stop.is_set():
                            interrupted = True
                            break
                        assignments = {}
                        for slot in range(slots):
                            if slot in batch.active_slots or exhausted or limit is not None and started >= limit:
                                continue
                            ticket = next(eligible, None)
                            if ticket is None:
                                exhausted = True
                                break
                            assignments[slot] = ticket
                            started += 1
                            restarted += ticket.rollout_id in known
                        if assignments:
                            update(batch.reset(assignments))
                        if not observations and not finished:
                            break
                        request = {"schema_version": 1, "type": "batch", "observations": [
                            {"slot": slot, "rollout_id": row["rollout"]["rollout_id"],
                             "sample_seed": str(row["rollout"]["sample_seed"]), "observation": row["observation"]}
                            for slot, row in sorted(observations.items())], "completed": finished}
                        if command is not None and owned_policy is None:
                            owned_policy = CommandBatchPolicy(command, session / "policy", timeout=response_seconds)
                        callback = owned_policy if owned_policy is not None else policy
                        actions, annotations = _responses(callback(json_copy(request)), set(observations))
                        batches += bool(observations)
                        finished = []
                        if stop is not None and stop.is_set():
                            interrupted = True
                            break
                        if actions:
                            update(batch.step(actions, policy_data=annotations))
                        if finished:
                            atomic_json(root / "collection-summary.json", _report(
                                root, selected, sampler.total, started=started, restarted=restarted,
                                skipped=len(completed), batches=batches))
                        if not observations and (exhausted or limit is not None and started >= limit) and not finished:
                            break
                except KeyboardInterrupt:
                    interrupted = True
                except Exception as exc:
                    batch.close_slots(batch.active_slots, status="error",
                                      reason=f"Batch policy error: {type(exc).__name__}: {exc}")
                    raise
        finally:
            if owned_policy is not None:
                owned_policy.close()
            selected, _, history = _snapshot(root)
            _write_selection(root, selected)
            report = _report(root, selected, sampler.total, started=started, restarted=restarted,
                             skipped=len(completed), batches=batches, interrupted=interrupted)
            report["attempt_history"] = history
            atomic_json(root / "collection-summary.json", report)
        return report
