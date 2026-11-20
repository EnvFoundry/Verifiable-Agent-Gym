import asyncio
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

from agent_gym.reference import ReferenceAgent


@unittest.skipUnless(importlib.util.find_spec("mcp"), "install the mcp extra for transport tests")
class MCPTests(unittest.IsolatedAsyncioTestCase):
    async def test_stdio_discovery_parallel_episodes_and_verification(self):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        with tempfile.TemporaryDirectory(prefix="gym mcp ") as folder:
            params = StdioServerParameters(command=sys.executable, args=["-m", "agent_gym", "mcp", "--output", folder])
            async with stdio_client(params) as (reader, writer):
                async with ClientSession(reader, writer) as client:
                    await client.initialize()
                    tools = await client.list_tools()
                    self.assertEqual({t.name for t in tools.tools}, {"gym_tasks", "gym_open", "gym_action", "gym_close", "gym_capabilities",
                                                               "gym_suite", "gym_collect", "gym_collection", "gym_rollouts", "gym_export_rollouts"})

                    async def call(name, arguments):
                        result = await client.call_tool(name, arguments)
                        self.assertFalse(result.isError, result)
                        return result.structuredContent or json.loads(result.content[0].text)

                    families = (await call("gym_tasks", {}))["families"]
                    async def solve(family):
                        observation = await call("gym_open", {"family": family, "seed": 17, "difficulty": 3})
                        episode_id, policy = observation["episode_id"], ReferenceAgent()
                        while observation["type"] != "terminal":
                            action = policy.act(observation)
                            observation = await call("gym_action", {"episode_id": episode_id, "action": action})
                        self.assertEqual(observation["result"]["status"], "passed")
                        self.assertTrue(Path(observation["result_directory"]).is_dir())
                    await asyncio.gather(*(solve(family) for family in families))
                    abandoned = await call("gym_open", {"family": families[0]})
                    closed = await call("gym_close", {"episode_id": abandoned["episode_id"]})
                    self.assertEqual(closed["status"], "cancelled")
                    self.assertIsNone(closed["reward"])

    async def test_stdio_collection_guidance_and_paged_outcomes(self):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        with tempfile.TemporaryDirectory(prefix="gym collection mcp ") as folder:
            params = StdioServerParameters(command=sys.executable, args=["-m", "agent_gym", "mcp", "--output", folder])
            async with stdio_client(params) as (reader, writer):
                async with ClientSession(reader, writer) as client:
                    await client.initialize()

                    async def call(name, arguments):
                        result = await client.call_tool(name, arguments)
                        self.assertFalse(result.isError, result)
                        return result.structuredContent or json.loads(result.content[0].text)

                    resources = await client.list_resources()
                    self.assertEqual(len(resources.resources), 4)
                    guide = await client.read_resource("gym://guides/grouped-collection")
                    self.assertIn("suite", guide.contents[0].text.lower())
                    prompts = await client.list_prompts()
                    self.assertEqual({prompt.name for prompt in prompts.prompts},
                                     {"design_task", "collect_experience", "compare_policies"})
                    designed = await client.get_prompt("design_task", {"capability": "parameter estimation",
                                                                       "population": "RC circuits", "outcome": "identified response"})
                    self.assertTrue(designed.messages)
                    capabilities = await call("gym_capabilities", {})
                    self.assertEqual(capabilities["page_limit"], 100)
                    await call("gym_suite", {"output": "suite.json", "name": "mcp-collection",
                                             "tasks": [{"family": "rc-identification", "seed": 1},
                                                       {"family": "rc-identification", "seed": 2}],
                                             "attempts_per_task": 2})
                    arguments = {"suite": "suite.json", "output": "collection", "policy_name": "reference",
                                 "policy_revision": "1", "reference": True, "slots": 2}
                    first = await call("gym_collect", {**arguments, "limit": 1})
                    self.assertEqual(first["completed"], 1)
                    resumed = await call("gym_collect", {**arguments, "resume": True})
                    self.assertEqual(resumed["completed"], 4)
                    state = await call("gym_collection", {"run": "collection"})
                    self.assertEqual(state["summary"]["remaining"], 0)
                    page = await call("gym_rollouts", {"run": "collection", "limit": 2})
                    self.assertEqual((page["total"], len(page["items"]), page["next_offset"]), (4, 2, 2))
                    exported = await call("gym_export_rollouts", {"run": "collection", "output": "trajectories.jsonl"})
                    self.assertEqual(exported["episodes"], 4)
