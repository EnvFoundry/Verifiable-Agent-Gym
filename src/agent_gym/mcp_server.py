"""MCP is an optional transport, not a second implementation of the Gym."""

from contextlib import asynccontextmanager
import asyncio
from pathlib import Path
import threading
from typing import Any, Literal

from .service import GymService


def create_server(output: Path, *, max_active: int = 32, episode_seconds: float = 600):
    try:
        from mcp.server.fastmcp import FastMCP
        from mcp.types import ToolAnnotations
    except ImportError as exc:
        raise RuntimeError('Install MCP support with: python -m pip install ".[mcp]"') from exc
    from .collection import ReferenceBatchPolicy, collect, collection_info, collection_rollouts
    from .contracts import TaskSpec
    from .guidance import GUIDES, collect_experience, compare_policies, design_task, get_guide
    from .rollout import PolicyVersion, export_rollouts
    from .suite import TaskSuite

    service = GymService(output, max_active=max_active, episode_seconds=episode_seconds)
    workspace = Path(output).resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    def path(value: str) -> Path:
        candidate = (workspace / value).resolve()
        if Path(value).is_absolute() or not candidate.is_relative_to(workspace):
            raise ValueError("use a path relative to the configured Gym output workspace")
        return candidate

    read = ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)
    write = ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=False)
    execute = ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=True)

    @asynccontextmanager
    async def lifespan(_server):
        try:
            yield service
        finally:
            await asyncio.to_thread(service.close_all)

    server = FastMCP("Verifiable Agent Gym", lifespan=lifespan, log_level="WARNING", instructions=(
        "List task families, open an episode, act through its tools, then submit a result. "
        "A gym_action tool call is {tool: name, arguments: {...}}; a final action is {submit: {...}}. "
        "The returned task prompt defines the acceptance criteria. Close episodes you abandon. "
        "For policy research, save a task suite and collect batched attempts using an external policy command. "
        "Collection paths are relative to the configured output workspace. Resume keeps completed outcomes "
        "and restarts interrupted attempts in fresh environments. Read compact progress with gym_collection "
        "and selected outcomes with gym_rollouts; export trajectories only when a consumer needs them. "
        "Research guides are resources under gym://guides/."
    ))

    @server.tool(annotations=read)
    async def gym_tasks() -> dict[str, Any]:
        """List available task families, dataset splits, and difficulty levels."""
        return await asyncio.to_thread(service.tasks)

    @server.tool(annotations=execute)
    async def gym_open(family: str, seed: int = 0, split: str = "train", difficulty: int = 1, version: int = 1) -> dict[str, Any]:
        """Create an independent episode and return its task, tools, and action budget."""
        return await asyncio.to_thread(service.open, family, seed, split, difficulty, version)

    @server.tool(annotations=execute)
    async def gym_action(episode_id: str, action: dict[str, Any]) -> dict[str, Any]:
        """Call one environment tool or submit the final answer for independent verification."""
        return await asyncio.to_thread(service.action, episode_id, action)

    @server.tool(annotations=write)
    async def gym_close(episode_id: str) -> dict[str, Any]:
        """Close an unfinished episode without assigning a false failure reward."""
        return await asyncio.to_thread(service.close, episode_id)

    @server.tool(annotations=read)
    async def gym_capabilities() -> dict[str, Any]:
        """Discover episode/collection operations, installed task families and focused research guides."""
        return service.capabilities() | {"workspace": str(workspace),
                                         "resources": [f"gym://guides/{topic}" for topic in GUIDES],
                                         "prompts": ["design_task", "collect_experience", "compare_policies"]}

    @server.tool(annotations=write)
    async def gym_suite(output: str, name: str, tasks: list[dict[str, Any]],
                        attempts_per_task: int = 1, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        """Save a finite task population and repeated-attempt design for collection or policy comparison."""
        suite = TaskSuite([TaskSpec(**spec) for spec in tasks], attempts_per_task=attempts_per_task,
                          name=name, metadata=metadata)
        destination = path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        suite.save(destination)
        return {"name": name, "tasks": len(suite.specs), "attempts_per_task": attempts_per_task,
                "output": output}

    @server.tool(annotations=execute)
    async def gym_collect(suite: str, output: str, policy_name: str, policy_revision: str,
                          command: list[str] | None = None, reference: bool = False,
                          generation: dict[str, Any] | None = None, slots: int = 4,
                          workers: int | None = None, seed: int = 0, epoch: int = 0,
                          rank: int = 0, world_size: int = 1, episode_seconds: float = 120,
                          response_seconds: float = 60, resume: bool = False,
                          retry_errors: bool = False, limit: int | None = None) -> dict[str, Any]:
        """Collect batched policy attempts; resume completed work and optionally limit new starts."""
        policy = ReferenceBatchPolicy() if reference else None
        stop = threading.Event()
        operation = asyncio.create_task(asyncio.to_thread(
                collect, path(suite), path(output), policy,
                policy_version=PolicyVersion(policy_name, policy_revision, generation), command=command,
                slots=slots, workers=workers, seed=seed, epoch=epoch, rank=rank, world_size=world_size,
                episode_seconds=episode_seconds, response_seconds=response_seconds, resume=resume,
                retry_errors=retry_errors, limit=limit, stop=stop))
        try:
            return await asyncio.shield(operation)
        except asyncio.CancelledError:
            stop.set()
            await operation
            raise
        finally:
            if policy is not None:
                policy.close()

    @server.tool(annotations=read)
    async def gym_collection(run: str) -> dict[str, Any]:
        """Read a collection's task population, policy identity and compact progress."""
        return await asyncio.to_thread(collection_info, path(run))

    @server.tool(annotations=read)
    async def gym_rollouts(run: str, offset: int = 0, limit: int = 20,
                           status: Literal["passed", "failed", "truncated", "error", "cancelled"] | None = None) -> dict[str, Any]:
        """Read up to 100 selected rollout outcomes; action histories remain in the trajectory export."""
        return await asyncio.to_thread(collection_rollouts, path(run), offset=offset, limit=limit, status=status)

    @server.tool(annotations=write)
    async def gym_export_rollouts(run: str, output: str, scored_only: bool = False) -> dict[str, Any]:
        """Export one selected episode per logical rollout, preserving groups and supplied policy statistics."""
        count = await asyncio.to_thread(export_rollouts, path(run), path(output), include_unscored=not scored_only)
        return {"episodes": count, "output": output}

    def register_guide(topic):
        def read_guide() -> str:
            return get_guide(topic)["content"]
        read_guide.__name__ = topic.replace("-", "_")
        server.resource(f"gym://guides/{topic}")(read_guide)

    for topic in GUIDES:
        register_guide(topic)
    for prompt in (design_task, collect_experience, compare_policies):
        server.prompt()(prompt)
    return server
