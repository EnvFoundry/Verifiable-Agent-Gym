"""Collect 16 real RC episodes through a persistent batched-policy process."""

import argparse
import json
from pathlib import Path
import sys
import time

from agent_gym import PolicyVersion, TaskSpec, TaskSuite, collect, export_rollouts
from agent_gym.collection import ReferenceBatchPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--policy", action="store_true", help="serve the scripted batch policy on stdin/stdout")
    args = parser.parse_args()
    if args.policy:
        policy = ReferenceBatchPolicy()
        try:
            for line in sys.stdin:
                print(json.dumps(policy(json.loads(line)), allow_nan=False), flush=True)
        finally:
            policy.close()
        return
    if args.output is None:
        parser.error("--output is required for collection")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    suite = TaskSuite([TaskSpec("rc-identification", seed=i) for i in range(4)],
                      attempts_per_task=4, name="rc-batched-policy")
    suite.save(root / "suite.json")
    command = [sys.executable, str(Path(__file__).resolve()), "--policy"]
    version = PolicyVersion("scripted-reference", "1", {"mechanism": "noisy-response-estimation"})
    started = time.perf_counter()
    first = collect(suite, root / "collection", policy_version=version,
                    command=command, slots=4, seed=17, limit=5)
    resumed = collect(root / "suite.json", root / "collection", policy_version=version,
                      command=command, slots=4, seed=17, resume=True)
    count = export_rollouts(root / "collection", root / "trajectories.jsonl")
    if first["completed"] != 5 or resumed["completed"] != 16 or count != 16:
        raise RuntimeError("collection did not preserve the requested sixteen logical attempts")
    print(json.dumps({"initial_completed": first["completed"], "resumed_completed": resumed["completed"],
                      "reused": resumed["skipped_completed"], "exported": count,
                      "counts": resumed["counts"], "elapsed_seconds": time.perf_counter() - started,
                      "output": str(root)}, indent=2))


if __name__ == "__main__":
    main()
