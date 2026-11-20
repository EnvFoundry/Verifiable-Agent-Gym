"""Shared episode operations for direct use, JSON lines, and MCP."""

from __future__ import annotations

from pathlib import Path
import math
import threading

from .contracts import TaskSpec
from .episode import Episode
from .registry import make, task_names


def capabilities() -> dict:
    from .guidance import GUIDES
    return {"schema_version": 1, "families": task_names(),
            "operations": {"episodes": ["tasks", "open", "action", "close"],
                           "collection": ["suite", "collect", "collection", "rollouts", "export-rollouts"]},
            "transports": ["python", "cli", "jsonl", "mcp"],
            "guides": list(GUIDES), "page_limit": 100,
            "batch_policy": {"request": "observations + completed", "response": "actions", "protocol": "jsonl-v1"}}


class GymService:
    def __init__(self, output: Path, *, max_active: int = 32, episode_seconds: float = 600):
        if max_active < 1 or not math.isfinite(episode_seconds) or episode_seconds <= 0:
            raise ValueError("max_active and episode_seconds must be positive")
        self.output = Path(output).resolve()
        self.max_active, self.seconds = max_active, episode_seconds
        self.episodes: dict[str, Episode] = {}
        self.lock = threading.RLock()

    def capabilities(self):
        return capabilities()

    def tasks(self):
        return {"schema_version": 1, "families": task_names(), "splits": ["train", "validation", "test"], "difficulties": [1, 2, 3]}

    def open(self, family: str, seed: int = 0, split: str = "train", difficulty: int = 1, version: int = 1):
        spec = TaskSpec(family, seed, split, difficulty, version)
        with self.lock:
            # Reap timed-out idle sessions so they do not occupy capacity forever.
            for key, episode in list(self.episodes.items()):
                if episode.remaining_seconds <= 0:
                    episode.stop("truncated", "Idle episode reached its time limit")
                    del self.episodes[key]
            if len(self.episodes) >= self.max_active:
                raise ValueError(f"All {self.max_active} active slots are in use; finish or close an episode")
            episode = Episode(make(spec), self.output, seconds=self.seconds)
            self.episodes[episode.id] = episode
            return episode.initial

    def action(self, episode_id: str, action: dict):
        with self.lock:
            episode = self.episodes.get(episode_id)
        if episode is None:
            raise ValueError("Episode is not active; use the ID returned by gym_open")
        response = episode.step(action)
        if episode.result is not None:
            with self.lock:
                self.episodes.pop(episode_id, None)
            response["result_directory"] = str(episode.path)
        return response

    def close(self, episode_id: str):
        with self.lock:
            episode = self.episodes.pop(episode_id, None)
        if episode is None:
            raise ValueError("Episode is not active")
        return episode.stop()

    def close_all(self):
        with self.lock:
            episodes = list(self.episodes.values())
            self.episodes.clear()
        for episode in episodes:
            episode.stop("cancelled", "Gym service stopped")


def serve_jsonl(service: GymService, input_stream, output_stream):
    import json
    from .storage import dumps
    try:
        for line in input_stream:
            request = {}
            try:
                request = json.loads(line)
                if not isinstance(request, dict):
                    raise ValueError("Request must be an object")
                methods = {"capabilities": service.capabilities, "tasks": service.tasks, "open": service.open, "action": service.action, "close": service.close}
                method = request.get("method")
                if method not in methods:
                    raise ValueError("method must be capabilities, tasks, open, action, or close")
                result = methods[method](**request.get("params", {}))
                response = {"id": request.get("id"), "result": result}
            except Exception as exc:
                response = {"id": request.get("id") if isinstance(request, dict) else None, "error": {"type": type(exc).__name__, "message": str(exc)}}
            output_stream.write(dumps(response) + "\n")
            output_stream.flush()
    finally:
        service.close_all()
