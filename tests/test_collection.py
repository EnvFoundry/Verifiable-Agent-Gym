import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest

from agent_gym import PolicyVersion, Task, TaskSpec, TaskSuite, Tool, Verdict, collect, export_rollouts
from agent_gym.collection import collection_info, collection_rollouts


ROOT = Path(__file__).resolve().parents[1]


class CountingWorld:
    tools = (Tool("observe", "Observe the current count", {"type": "object"}),)

    def __init__(self):
        self.calls = 0
        self.closed = False

    def call(self, tool, arguments):
        self.calls += 1
        return {"count": self.calls}

    def close(self):
        self.closed = True


def factory(spec):
    world = CountingWorld()
    return Task(spec, "Return the number of calls", world,
                lambda answer: Verdict(answer == {"count": world.calls}, float(answer == {"count": world.calls}), "count"))


def one_step(request):
    return {"actions": [{"slot": row["slot"], "action": {"submit": {"count": 0}},
                         "policy_data": {"tokens": 3}}
                        for row in request["observations"]]}


def cli_environment():
    return dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONIOENCODING="utf-8")


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="gym collection α ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.suite = TaskSuite([TaskSpec("counting-world", seed=i) for i in range(4)], attempts_per_task=3)
        self.policy = PolicyVersion("counting-policy", "1", {"temperature": .7})

    def run_collection(self, callback=one_step, **kwargs):
        return collect(self.suite, self.root / "run", callback, policy_version=self.policy,
                       slots=3, task_factory=factory, **kwargs)

    def test_chunk_resume_reuses_finished_episodes_and_unique_exports(self):
        first = self.run_collection(limit=4)
        paths = {path: path.read_bytes() for path in (self.root / "run").glob("sessions/*/*/*/result.json")}
        resumed = self.run_collection(resume=True)
        self.assertEqual((first["completed"], resumed["completed"], resumed["skipped_completed"]), (4, 12, 4))
        self.assertEqual(resumed["started_this_session"], 8)
        for path, data in paths.items():
            self.assertEqual(path.read_bytes(), data)
        target = self.root / "rollouts.jsonl"
        self.assertEqual(export_rollouts(self.root / "run", target), 12)
        rows = [json.loads(line) for line in target.read_text().splitlines()]
        self.assertEqual(len({row["rollout"]["rollout_id"] for row in rows}), 12)
        self.assertEqual(len({row["rollout"]["group_id"] for row in rows}), 4)
        for row in rows:
            self.assertEqual(row["policy_steps"], [{"step": 1, "data": {"tokens": 3}}])
            self.assertTrue((self.root / "run" / row["rollout"]["path"] / "result.json").is_file())
        no_work = self.run_collection(resume=True)
        self.assertEqual(no_work["started_this_session"], 0)
        self.assertEqual(no_work["policy_batches"], 0)

    def test_variable_length_episodes_refill_free_slots(self):
        seen, completed, batches = {}, [], []
        def policy(request):
            completed.extend(request["completed"])
            batches.append([(row["slot"], row["rollout_id"]) for row in request["observations"]])
            actions = []
            for row in request["observations"]:
                self.assertIsInstance(row["sample_seed"], str)
                observation = row["observation"]
                if observation["type"] == "start":
                    seen[row["rollout_id"]] = observation["task"]["seed"]
                    target = seen[row["rollout_id"]]
                    action = {"submit": {"count": 0}} if target == 0 else {"tool": "observe", "arguments": {}}
                else:
                    count = observation["result"]["count"]
                    action = {"submit": {"count": count}} if count == seen[row["rollout_id"]] else {"tool": "observe", "arguments": {}}
                actions.append({"slot": row["slot"], "action": action})
            return {"actions": actions}
        report = self.run_collection(policy, shuffle=False)
        self.assertEqual(report["counts"]["passed"], 12)
        self.assertEqual(len(completed), 12)
        self.assertFalse(batches[-1])
        self.assertGreater(len({identifier for batch in batches for slot, identifier in batch if slot == 0}), 1)

    def test_interrupt_restarts_only_unfinished_tickets(self):
        calls = [0]
        def interrupted(request):
            calls[0] += 1
            if calls[0] == 2:
                raise KeyboardInterrupt
            return one_step(request)
        first = self.run_collection(interrupted)
        self.assertTrue(first["interrupted"])
        self.assertEqual(first["completed"], 3)
        self.assertEqual(first["counts"]["cancelled"], 3)
        first_paths = list((self.root / "run").glob("sessions/*/*/*/result.json"))
        old_cancelled = [p for p in first_paths if json.loads(p.read_text())["status"] == "cancelled"]
        resumed = self.run_collection(resume=True)
        self.assertEqual(resumed["counts"]["passed"], 12)
        self.assertEqual(resumed["restarted_this_session"], 3)
        self.assertEqual(resumed["counts"]["cancelled"], 0)
        self.assertEqual(resumed["attempt_history"]["attempts"], 15)
        self.assertEqual(resumed["attempt_history"]["counts"]["cancelled"], 3)
        self.assertTrue(all(json.loads(p.read_text())["status"] == "cancelled" for p in old_cancelled))
        exported = self.root / "unique.jsonl"
        self.assertEqual(export_rollouts(self.root / "run", exported), 12)

    def test_recovery_reads_terminal_files_even_without_selection_index(self):
        report = self.run_collection(limit=4)
        (self.root / "run/rollout-index.jsonl").unlink()
        (self.root / "run/collection-summary.json").unlink()
        for index in (self.root / "run").glob("sessions/*/rollout-index.jsonl"):
            index.write_text("", encoding="utf-8")
        resumed = self.run_collection(resume=True)
        self.assertEqual(report["completed"], resumed["skipped_completed"])
        self.assertEqual(resumed["counts"]["passed"], 12)

    def test_cooperative_stop_retains_restartable_attempts(self):
        stop = threading.Event()
        def stopping(request):
            stop.set()
            return one_step(request)
        report = self.run_collection(stopping, stop=stop)
        self.assertTrue(report["interrupted"])
        self.assertEqual(report["completed"], 0)
        self.assertEqual(report["counts"]["cancelled"], 3)
        for result_path in (self.root / "run").glob("sessions/*/*/*/result.json"):
            self.assertEqual(json.loads(result_path.read_text())["steps"], 0)
        resumed = self.run_collection(resume=True)
        self.assertEqual(resumed["completed"], 12)
        self.assertEqual(resumed["restarted_this_session"], 3)

    def test_transport_errors_are_unscored_and_explicitly_retryable(self):
        def broken(request):
            raise RuntimeError("inference service unavailable")
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            self.run_collection(broken)
        failed = collection_info(self.root / "run")["summary"]
        self.assertEqual(failed["counts"]["error"], 3)
        continued = self.run_collection(resume=True)
        self.assertEqual(continued["counts"]["error"], 3)
        self.assertEqual(continued["counts"]["passed"], 9)
        retried = self.run_collection(resume=True, retry_errors=True)
        self.assertEqual(retried["counts"]["passed"], 12)
        self.assertEqual(retried["restarted_this_session"], 3)
        self.assertEqual(retried["attempt_history"]["counts"]["error"], 3)
        self.assertEqual(retried["attempt_history"]["attempts"], 15)

    def test_batch_schema_error_does_not_partially_apply_actions(self):
        def bad_batch(request):
            rows = one_step(request)["actions"]
            return {"actions": rows + rows[:1]}
        with self.assertRaisesRegex(ValueError, "exactly once"):
            self.run_collection(bad_batch)
        for path in (self.root / "run").glob("sessions/*/*/*/result.json"):
            value = json.loads(path.read_text())
            self.assertEqual(value["steps"], 0)
            self.assertEqual(value["status"], "error")
            self.assertIsNone(value["reward"])

    def test_resume_binds_population_policy_and_sampling(self):
        self.run_collection(limit=1)
        with self.assertRaisesRegex(ValueError, "policy"):
            collect(self.suite, self.root / "run", one_step, policy_version=PolicyVersion("different", "1"),
                    slots=3, task_factory=factory, resume=True)
        with self.assertRaisesRegex(ValueError, "sampling"):
            self.run_collection(resume=True, seed=2)
        with self.assertRaisesRegex(ValueError, "resume"):
            self.run_collection()

    def test_whole_task_rank_partition_and_empty_rank(self):
        roots = []
        for rank in range(3):
            path = self.root / f"rank-{rank}"
            report = collect(self.suite, path, one_step, policy_version=self.policy, slots=2,
                             task_factory=factory, rank=rank, world_size=3)
            self.assertEqual(report["remaining"], 0)
            roots.append(collection_rollouts(path, limit=100)["items"])
        task_sets = [{row["task"]["seed"] for row in rows} for rows in roots]
        self.assertFalse(task_sets[0] & task_sets[1] | task_sets[1] & task_sets[2] | task_sets[0] & task_sets[2])
        self.assertEqual(set.union(*task_sets), set(range(4)))
        self.assertEqual(sum(map(len, roots)), 12)
        empty = collect(self.suite, self.root / "empty", one_step, policy_version=self.policy,
                        slots=2, task_factory=factory, rank=7, world_size=8)
        self.assertEqual(empty["requested"], 0)
        self.assertEqual(empty["policy_batches"], 0)

    def test_page_filters_and_bounds(self):
        self.run_collection()
        page = collection_rollouts(self.root / "run", status="passed", offset=4, limit=3)
        self.assertEqual((page["total"], len(page["items"]), page["next_offset"]), (12, 3, 7))
        self.assertEqual(collection_rollouts(self.root / "run", status="failed")["items"], [])
        for kwargs in ({"limit": 101}, {"offset": -1}, {"limit": True}, {"status": "running"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                collection_rollouts(self.root / "run", **kwargs)

    def test_standalone_command_example(self):
        result = subprocess.run([sys.executable, str(ROOT / "examples/collect_policy.py"),
                                 "--output", str(self.root / "command")],
                                capture_output=True, text=True, encoding="utf-8", env=cli_environment(), timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        value = json.loads(result.stdout)
        self.assertEqual((value["initial_completed"], value["resumed_completed"], value["exported"]), (5, 16, 16))

    def test_installed_cli_suite_collection_and_resume(self):
        def cli(*args):
            result = subprocess.run([sys.executable, "-m", "agent_gym", *map(str, args)], cwd=self.root,
                                    capture_output=True, text=True, encoding="utf-8", env=cli_environment(), timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            return json.loads(result.stdout)
        suite_path = self.root / "suite.json"
        cli("suite", "--tasks", "rc-identification", "--count", 2, "--attempts", 2, "--output", suite_path)
        options = ("collect", suite_path, "--reference", "--policy-name", "scripted-reference", "--policy-revision", "1",
                   "--slots", 2, "--output", self.root / "cli")
        self.assertEqual(cli(*options, "--limit", 1)["completed"], 1)
        self.assertEqual(cli(*options, "--resume")["completed"], 4)
        self.assertEqual(cli("collection", self.root / "cli")["summary"]["remaining"], 0)
        self.assertEqual(len(cli("rollouts", self.root / "cli", "--limit", 2)["items"]), 2)


if __name__ == "__main__":
    unittest.main()
