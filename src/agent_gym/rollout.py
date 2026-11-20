"""Batched external-policy rollouts over the shared Episode implementation."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import math
from pathlib import Path
import threading
from typing import Callable, Iterable, Mapping

from .contracts import Json, Task, TaskSpec
from .episode import Episode
from .registry import make
from .storage import RunLease, atomic_json, dumps, json_copy
from .suite import RolloutTicket, fingerprint


@dataclass(frozen=True)
class PolicyVersion:
    name: str
    revision: str
    generation: Json | None = None

    def to_dict(self) -> Json:
        if any(not isinstance(value, str) or not value.strip() for value in (self.name, self.revision)):
            raise ValueError("policy name and revision must be nonempty")
        if self.generation is not None and not isinstance(self.generation, dict):
            raise ValueError("policy generation configuration must be an object")
        return json_copy({"name": self.name, "revision": self.revision, "generation": self.generation or {}})


@dataclass
class _Slot:
    episode: Episode
    rollout: Json


class RolloutBatch:
    """Independent episode slots driven by a trainer's own batched policy.

    reset/step accept mappings so completed slots can be refilled while other
    episodes continue. Separate slots execute concurrently; calls on one batch
    are serialized. The trainer supplies actions and optional policy-side data.
    """

    def __init__(self, root: str | Path, *, slots: int, policy: PolicyVersion,
                 workers: int | None = None, episode_seconds: float = 120,
                 task_factory: Callable[[TaskSpec], Task] = make):
        if type(slots) is not int or slots < 1:
            raise ValueError("slots must be a positive integer")
        workers = slots if workers is None else workers
        if type(workers) is not int or workers < 1:
            raise ValueError("workers must be a positive integer")
        if type(episode_seconds) not in (int, float) or not math.isfinite(episode_seconds) or episode_seconds <= 0:
            raise ValueError("episode_seconds must be finite and positive")
        self.policy = PolicyVersion(**policy.to_dict())
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.capacity, self.seconds = slots, episode_seconds
        self._factory = task_factory
        self._slots: dict[int, _Slot] = {}
        self._issued: set[str] = set()
        self._counts = Counter()
        self._lock = threading.RLock()
        self._closed = False
        self._lease = RunLease(self.root)
        self._lease.__enter__()
        try:
            if (self.root / "rollout-run.json").exists() or (self.root / "run.json").exists():
                raise ValueError("output already contains a run; use a new directory for a rollout batch")
            atomic_json(self.root / "rollout-run.json", {"schema_version": 1, "slots": slots,
                        "workers": workers, "episode_seconds": episode_seconds})
            self._index = (self.root / "rollout-index.jsonl").open("x", encoding="utf-8", newline="\n")
            self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="gym-rollout")
        except BaseException:
            self._lease.__exit__(None, None, None)
            raise

    @property
    def active_slots(self) -> tuple[int, ...]:
        with self._lock:
            return tuple(sorted(self._slots))

    def _validate_slots(self, values: Mapping[int, object], *, occupied: bool):
        if self._closed:
            raise ValueError("rollout batch is closed")
        if not isinstance(values, Mapping):
            raise ValueError("slot assignments must be a mapping")
        for slot in values:
            if type(slot) is not int or not 0 <= slot < self.capacity:
                raise ValueError("slot index is outside the configured batch")
            if (slot in self._slots) != occupied:
                raise ValueError(f"slot {slot} is {'already active' if not occupied else 'not active'}")

    def _finish_index(self, slot: int, record: Json, result: Json, *, setup_error: bool = False):
        self._counts[result["status"]] += 1
        self._index.write(dumps({"rollout_id": record["rollout_id"], "path": record["path"],
                                 "setup_error": setup_error}) + "\n")
        self._index.flush()
        self._slots.pop(slot, None)

    @staticmethod
    def _response(slot: int, record: Json, observation: Json, result: Json | None = None) -> Json:
        status = result["status"] if result else "running"
        scored = status in ("passed", "failed") and result["reward"] is not None
        return {"slot": slot, "observation": observation, "done": result is not None,
                "terminated": status in ("passed", "failed"), "truncated": status == "truncated",
                "scored": scored, "reward": result["reward"] if scored else None,
                "status": status, "rollout": json_copy(record), "result": result}

    def reset(self, assignments: Mapping[int, RolloutTicket], *, policy: PolicyVersion | None = None) -> dict[int, Json]:
        with self._lock:
            self._validate_slots(assignments, occupied=False)
            policy_data = (policy or self.policy).to_dict()
            tickets = list(assignments.values())
            if any(not isinstance(ticket, RolloutTicket) for ticket in tickets):
                raise ValueError("reset requires RolloutTicket values")
            identifiers = [ticket.rollout_id for ticket in tickets]
            if len(set(identifiers)) != len(identifiers) or any(value in self._issued for value in identifiers):
                raise ValueError("rollout tickets must be unique within a run")
            self._issued.update(identifiers)
            futures = {slot: self._pool.submit(self._open, ticket, policy_data) for slot, ticket in assignments.items()}
            responses = {}
            failures = []
            for slot, future in futures.items():
                try:
                    episode, record, error = future.result()
                except Exception as exc:
                    failures.append(exc)
                    continue
                if error is not None:
                    self._finish_index(slot, record, error, setup_error=True)
                    responses[slot] = self._response(slot, record, {"type": "terminal", "result": error}, error)
                else:
                    self._slots[slot] = _Slot(episode, record)
                    responses[slot] = self._response(slot, record, episode.initial)
            if failures:
                raise failures[0]
            return responses

    def _open(self, ticket: RolloutTicket, policy: Json):
        record = {**ticket.to_dict(), "task_group_id": ticket.group_id,
                  "group_id": fingerprint({"task_group_id": ticket.group_id, "policy": policy}),
                  "policy": policy}
        task = None
        episode = None
        try:
            task = self._factory(ticket.spec)
            if task.spec != ticket.spec:
                raise ValueError("task factory returned a different task specification")
            if hasattr(task, "data_reference"):
                record["data_reference"] = json_copy(task.data_reference)
            record["limits"] = {"actions": task.max_actions, "seconds": self.seconds}
            record["group_id"] = fingerprint({"task_group_id": ticket.group_id, "policy": policy,
                                               "data_reference": record.get("data_reference"),
                                               "limits": record["limits"]})
            episode = Episode(task, self.root, seconds=self.seconds, agent=policy["name"])
            record["episode_id"] = episode.id
            record["path"] = episode.path.relative_to(self.root).as_posix()
            atomic_json(episode.path / "rollout.json", record)
            return episode, record, None
        except Exception as exc:
            if episode is not None:
                episode.stop("error", f"Rollout setup error: {type(exc).__name__}: {exc}")
            elif task is not None:
                try:
                    task.environment.close()
                except Exception:
                    pass
            record["path"] = f"setup-errors/{ticket.rollout_id}.json"
            error = {"task": ticket.spec.to_dict(), "episode_id": None, "status": "error", "reward": None,
                     "verification": None, "agent": policy["name"], "reason": f"Setup error: {type(exc).__name__}: {exc}"}
            destination = self.root / record["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(destination, {"rollout": record, "result": error})
            return None, record, error

    def step(self, actions: Mapping[int, Json], *, policy_data: Mapping[int, Json] | None = None) -> dict[int, Json]:
        """Step selected slots; optional policy_data preserves caller-provided token statistics."""
        with self._lock:
            self._validate_slots(actions, occupied=True)
            copied = {slot: json_copy(action) for slot, action in actions.items()}
            if policy_data is not None and not isinstance(policy_data, Mapping):
                raise ValueError("policy_data must map slot indices to policy-side data")
            annotations = {} if policy_data is None else json_copy(dict(policy_data))
            # JSON object keys become strings; normalize only the known integer slots.
            if any(str(slot) not in {str(key) for key in actions} for slot in annotations):
                raise ValueError("policy_data must refer only to stepped slots")
            if any(not isinstance(value, dict) for value in annotations.values()):
                raise ValueError("per-action policy_data must be an object")
            futures = {slot: self._pool.submit(self._step, self._slots[slot], action, annotations.get(str(slot)))
                       for slot, action in copied.items()}
            responses = {}
            for slot, future in futures.items():
                observation = future.result()
                state = self._slots[slot]
                result = state.episode.result
                responses[slot] = self._response(slot, state.rollout, observation, result)
                if result is not None:
                    self._finish_index(slot, state.rollout, result)
            return responses

    def _step(self, state: _Slot, action: Json, policy_data: Json | None):
        previous = state.episode.steps
        observation = state.episode.step(action)
        if policy_data is not None and state.episode.steps > previous:
            with (state.episode.path / "policy-steps.jsonl").open("a", encoding="utf-8", newline="\n") as output:
                output.write(dumps({"step": state.episode.steps, "data": policy_data}) + "\n")
        return observation

    def close_slots(self, slots: Iterable[int], *, reason: str = "Closed by trainer",
                    status: str = "cancelled") -> dict[int, Json]:
        if status not in ("cancelled", "error", "truncated"):
            raise ValueError("close status must be cancelled, error, or truncated")
        with self._lock:
            values = list(slots)
            if len(values) != len(set(values)):
                raise ValueError("slot IDs must be unique")
            selected = {slot: None for slot in values}
            self._validate_slots(selected, occupied=True)
            futures = {slot: self._pool.submit(self._slots[slot].episode.stop, status, reason) for slot in values}
            responses = {}
            for slot, future in futures.items():
                result = future.result()
                state = self._slots[slot]
                responses[slot] = self._response(slot, state.rollout, {"type": "terminal", "result": result}, result)
                self._finish_index(slot, state.rollout, result)
            return responses

    def summary(self) -> Json:
        with self._lock:
            counts = {status: self._counts[status] for status in ("passed", "failed", "truncated", "error", "cancelled")}
            return {"completed": sum(counts.values()), "active": len(self._slots), "counts": counts}

    def close(self):
        with self._lock:
            if self._closed:
                return
            try:
                self.close_slots(self.active_slots, reason="Rollout batch closed")
                atomic_json(self.root / "rollout-summary.json", self.summary())
            finally:
                self._pool.shutdown(wait=True)
                self._index.close()
                self._lease.__exit__(None, None, None)
                self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def export_rollouts(root: str | Path, destination: str | Path, *, include_unscored: bool = True) -> int:
    """Stream completed episodes in export-v1 shape with factual rollout metadata."""
    root, destination = Path(root).resolve(), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("x", encoding="utf-8", newline="\n") as output:
        with (root / "rollout-index.jsonl").open(encoding="utf-8") as index:
            for line in index:
                entry = json.loads(line)
                if entry["setup_error"]:
                    # Setup failures have no episode or factual action trace.
                    # They remain in the run index/summary, not episode exports.
                    continue
                path = root / entry["path"]
                result = json.loads((path / "result.json").read_text(encoding="utf-8"))
                if not include_unscored and (result["status"] not in ("passed", "failed") or result["reward"] is None):
                    continue
                rollout = json.loads((path / "rollout.json").read_text(encoding="utf-8"))
                rollout["path"] = entry["path"]
                with (path / "events.jsonl").open(encoding="utf-8") as events:
                    trajectory = [event for event in map(json.loads, events) if event["kind"] in ("observation", "action")]
                policy_path = path / "policy-steps.jsonl"
                with policy_path.open(encoding="utf-8") if policy_path.exists() else _empty_lines() as steps:
                    policy_steps = list(map(json.loads, steps))
                record = {"schema_version": 1, "episode_id": result["episode_id"], "task": result["task"],
                          "agent": result["agent"], "status": result["status"], "reward": result["reward"],
                          "verification": result["verification"], "trajectory": trajectory,
                          "rollout": rollout, "policy_steps": policy_steps}
                output.write(dumps(record) + "\n")
                count += 1
    return count


def _empty_lines():
    from contextlib import nullcontext
    return nullcontext(())
