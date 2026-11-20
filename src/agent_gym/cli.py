from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from . import __version__
from .contracts import TaskSpec
from .registry import make, task_names
from .storage import dumps


def default_output() -> Path:
    configured = os.environ.get("AGENT_GYM_RUNS")
    return Path(configured) if configured else Path.home() / ".agent-gym" / "runs"


def parser():
    result = argparse.ArgumentParser(prog="agent-gym", description="Executable tasks and independent verification for tool-using agents")
    result.add_argument("--version", action="version", version=__version__)
    result.add_argument("--task-pack", action="append", default=[], metavar="FAMILY=PATH",
                        help="load a data-backed SQL task pack before running a command")
    sub = result.add_subparsers(dest="operation", required=True)
    sub.add_parser("list", help="List installed task families")
    for name in ("run", "generate", "suite"):
        command = sub.add_parser(name)
        command.add_argument("--tasks", nargs="+", default=["all"])
        command.add_argument("--count", type=int, help="Instances per family; defaults to 10 generated or all matching pack records")
        command.add_argument("--seed", type=int, default=0, help="First generator seed or finite-pack record index")
        command.add_argument("--split", choices=("train", "validation", "test"), default="train")
        command.add_argument("--difficulty", type=int, choices=(1, 2, 3), help="Generated difficulty (default 1), or pack record filter")
        command.add_argument("--output", type=Path, required=True)
        if name == "suite":
            command.add_argument("--attempts", type=int, default=1)
            command.add_argument("--name", default="task-suite")
        if name == "run":
            command.add_argument("--agent", choices=("reference", "empty", "command"), default="reference")
            command.add_argument("--command-json", help="Agent executable and arguments as a JSON array; use absolute paths")
            command.add_argument("--jobs", type=int, default=4)
            command.add_argument("--episode-seconds", type=float, default=120)
            command.add_argument("--action-seconds", type=float, default=30)
            command.add_argument("--resume", action="store_true")
            command.add_argument("--retry-errors", action="store_true")
            command.add_argument("--require-pass", action="store_true")
    command = sub.add_parser("collect", help="Collect a task suite through a batched policy, with continuation")
    command.add_argument("suite", type=Path)
    choice = command.add_mutually_exclusive_group(required=True)
    choice.add_argument("--command-json", help="One persistent batched-policy process, explicit JSON argv")
    choice.add_argument("--reference", action="store_true", help="Use the bundled observation-only reference policy")
    command.add_argument("--policy-name", required=True)
    command.add_argument("--policy-revision", required=True)
    command.add_argument("--generation", default="{}", help="Policy decoding configuration as JSON")
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--slots", type=int, default=4)
    command.add_argument("--workers", type=int)
    command.add_argument("--seed", type=int, default=0)
    command.add_argument("--epoch", type=int, default=0)
    command.add_argument("--rank", type=int, default=0)
    command.add_argument("--world-size", type=int, default=1)
    command.add_argument("--episode-seconds", type=float, default=120)
    command.add_argument("--response-seconds", type=float, default=60)
    command.add_argument("--resume", action="store_true")
    command.add_argument("--retry-errors", action="store_true")
    command.add_argument("--limit", type=int, help="New rollouts to start in this invocation; drain before returning")
    command = sub.add_parser("collection", help="Read a collection's population, policy and progress")
    command.add_argument("path", type=Path)
    command = sub.add_parser("rollouts", help="Read a bounded page of selected collection outcomes")
    command.add_argument("path", type=Path)
    command.add_argument("--offset", type=int, default=0)
    command.add_argument("--limit", type=int, default=20)
    command.add_argument("--status", choices=("passed", "failed", "truncated", "error", "cancelled"))
    command = sub.add_parser("guide", help="Read a focused research and operating guide")
    command.add_argument("topic", nargs="?", default="task-design")
    sub.add_parser("capabilities", help="Discover operations, transports and research guides")
    for name in ("serve", "mcp"):
        command = sub.add_parser(name, help="Serve episodes over " + ("JSON lines" if name == "serve" else "MCP stdio"))
        command.add_argument("--output", type=Path, default=default_output())
        command.add_argument("--max-active", type=int, default=32)
        command.add_argument("--episode-seconds", type=float, default=600)
    command = sub.add_parser("export", help="Export observed actions, feedback, and rewards as JSONL")
    command.add_argument("run", type=Path)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--include-unscored", action="store_true")
    command = sub.add_parser("export-rollouts", help="Export batched trainer episodes with group and policy identity")
    command.add_argument("run", type=Path)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--scored-only", action="store_true")
    command = sub.add_parser("pack-info", help="Inspect a data-backed SQL task pack")
    command.add_argument("path", type=Path)
    command.add_argument("--family", default="sql-dataset")
    command = sub.add_parser("connect", help="Write MCP or Research OS configuration for this installed interpreter")
    command.add_argument("--host", choices=("mcp", "research-os"), default="mcp")
    command.add_argument("--output", type=Path, required=True)
    return result


def main(argv=None):
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = parser().parse_args(argv)
    try:
        pack_arguments = []
        packs = {}
        if args.task_pack:
            from .taskpacks import load_taskpack
            for entry in args.task_pack:
                if "=" not in entry:
                    raise ValueError("--task-pack expects FAMILY=PATH")
                family, raw_path = entry.split("=", 1)
                path = Path(raw_path).resolve(strict=True)
                packs[family] = load_taskpack(path, family=family).register()
                pack_arguments.extend(["--task-pack", f"{family}={path}"])
        if args.operation == "list":
            print("\n".join(task_names()))
        elif args.operation in ("run", "generate", "suite"):
            if args.count is not None and args.count < 1:
                raise ValueError("count must be positive")
            if args.seed < 0:
                raise ValueError("seed must be nonnegative")
            families = task_names() if args.tasks == ["all"] else args.tasks
            unknown = set(families) - set(task_names())
            if unknown:
                raise ValueError(f"Unknown tasks: {', '.join(sorted(unknown))}")
            specs = []
            for family in families:
                if family in packs:
                    available = [spec for spec in packs[family].specs(args.split)
                                 if spec.seed >= args.seed and
                                 (args.difficulty is None or spec.difficulty == args.difficulty)]
                    if not available or (args.count is not None and args.count > len(available)):
                        raise ValueError(f"{family}: requested selection exceeds the {len(available)} matching pack records")
                    specs.extend(available if args.count is None else available[:args.count])
                else:
                    specs.extend(TaskSpec(family, args.seed + index, args.split, args.difficulty or 1)
                                 for index in range(args.count or 10))
            if args.operation == "suite":
                from .suite import TaskSuite
                suite = TaskSuite(specs, attempts_per_task=args.attempts, name=args.name,
                                  metadata={"taskpacks": {family: pack.describe() for family, pack in packs.items()}})
                args.output.parent.mkdir(parents=True, exist_ok=True)
                suite.save(args.output)
                print(json.dumps({"name": args.name, "tasks": len(specs), "attempts_per_task": args.attempts,
                                  "output": str(args.output)}, indent=2))
            elif args.operation == "generate":
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with args.output.open("x", encoding="utf-8", newline="\n") as stream:
                    for spec in specs:
                        task = make(spec)
                        try:
                            stream.write(dumps({"schema_version": 1, "spec": spec.to_dict(), "observation": task.observation()}) + "\n")
                        finally:
                            task.environment.close()
                print(f"Generated {len(specs)} tasks: {args.output}")
            else:
                from .process import CommandAgent
                from .reference import EmptyAgent, ReferenceAgent
                from .runner import run_batch
                config = {}
                if args.agent == "command":
                    if not args.command_json:
                        raise ValueError("--agent command requires --command-json")
                    command = json.loads(args.command_json)
                    if not isinstance(command, list) or not command or not all(isinstance(x, str) and x for x in command):
                        raise ValueError("--command-json must be a nonempty JSON array of strings")
                    factory = lambda path: CommandAgent(command, path)
                    config = {"argv": command}
                else:
                    if args.command_json:
                        raise ValueError("--command-json requires --agent command")
                    factory = lambda path: ReferenceAgent() if args.agent == "reference" else EmptyAgent()
                report = run_batch(specs, args.output, factory, agent_name=args.agent, agent_config=config, jobs=args.jobs,
                                   seconds=args.episode_seconds, action_seconds=args.action_seconds, resume=args.resume, retry_errors=args.retry_errors)
                print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))
                if report["interrupted"]:
                    return 130
                if report["counts"]["error"] or (args.require_pass and report["counts"]["passed"] != report["requested"]):
                    return 1
        elif args.operation == "collect":
            from .collection import ReferenceBatchPolicy, collect
            from .rollout import PolicyVersion
            reference = ReferenceBatchPolicy() if args.reference else None
            try:
                report = collect(args.suite, args.output, reference,
                                 policy_version=PolicyVersion(args.policy_name, args.policy_revision, json.loads(args.generation)),
                                 command=json.loads(args.command_json) if args.command_json else None,
                                 slots=args.slots, workers=args.workers, seed=args.seed, epoch=args.epoch,
                                 rank=args.rank, world_size=args.world_size, episode_seconds=args.episode_seconds,
                                 response_seconds=args.response_seconds, resume=args.resume, retry_errors=args.retry_errors,
                                 limit=args.limit)
            finally:
                if reference is not None:
                    reference.close()
            print(json.dumps(report, ensure_ascii=False, indent=2))
            if report["interrupted"]:
                return 130
            if report["counts"]["error"]:
                return 1
        elif args.operation == "collection":
            from .collection import collection_info
            print(json.dumps(collection_info(args.path), ensure_ascii=False, indent=2))
        elif args.operation == "rollouts":
            from .collection import collection_rollouts
            print(json.dumps(collection_rollouts(args.path, offset=args.offset, limit=args.limit,
                                                 status=args.status), ensure_ascii=False, indent=2))
        elif args.operation == "guide":
            from .guidance import get_guide
            print(json.dumps(get_guide(args.topic), ensure_ascii=False, indent=2))
        elif args.operation == "capabilities":
            from .service import capabilities
            print(json.dumps(capabilities(), ensure_ascii=False, indent=2))
        elif args.operation in ("serve", "mcp"):
            if args.operation == "mcp":
                from .mcp_server import create_server
                create_server(args.output, max_active=args.max_active, episode_seconds=args.episode_seconds).run(transport="stdio")
            else:
                from .service import GymService, serve_jsonl
                serve_jsonl(GymService(args.output, max_active=args.max_active, episode_seconds=args.episode_seconds), sys.stdin, sys.stdout)
        elif args.operation == "export":
            from .export import export_run
            count = export_run(args.run, args.output, include_unscored=args.include_unscored)
            print(f"Exported {count} episodes: {args.output}")
        elif args.operation == "export-rollouts":
            from .rollout import export_rollouts
            count = export_rollouts(args.run, args.output, include_unscored=not args.scored_only)
            print(f"Exported {count} episodes: {args.output}")
        elif args.operation == "pack-info":
            from .taskpacks import load_taskpack
            print(json.dumps(load_taskpack(args.path, family=args.family).describe(), ensure_ascii=False, indent=2))
        elif args.operation == "connect":
            from importlib.util import find_spec
            if find_spec("mcp") is None:
                raise RuntimeError('Install MCP support first: python -m pip install ".[mcp]"')
            server = {"command": sys.executable, "args": ["-m", "agent_gym", *pack_arguments, "mcp"]}
            if args.host == "research-os":
                server["transport"] = "stdio"
                value = {"name": "verifiable-agent-gym", "version": __version__, "description": "Executable tasks and independent verification", "mcpServers": {"agent-gym": server}}
            else:
                value = {"mcpServers": {"agent-gym": server}}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(value, stream, indent=2)
                stream.write("\n")
            print(f"Wrote {args.output}. This configuration uses the installed interpreter {sys.executable}.")
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"agent-gym: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
