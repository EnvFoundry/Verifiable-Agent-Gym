"""Installed guidance for task construction, interaction and policy research."""

from __future__ import annotations

import json


GUIDES = {
    "task-design": (
        "Construct a task with observable progress and meaningful outcomes",
        """Begin with a capability and a population of situations where it matters.
Define what the agent initially knows, what each action can reveal or change, and
the artifact or outcome that completes the task. Preserve units, timing, resource
limits and state transitions when they affect the scientific question.

Keep task generation, environment behavior and result evaluation distinct. A
verifier should accept valid alternative solutions, not only reproduce one policy's
trajectory. A SQL task can accept different queries with the same result; a physical
inverse problem can compare predicted responses independently of the agent's fitting
method. Score the intended observable outcome and retain explanatory measurements.

Use TaskSpec for instances and a task package for an existing domain engine or
dataset. Generated seeds sample instances of the chosen mechanism; more seeds do
not by themselves establish transfer to a new mechanism. For finite packs, enumerate
the available specifications in the requested split rather than guessing seed IDs.
Group related variants when constructing a suite for policy comparisons.

Deliver a working environment, an example interaction and clear outcome semantics.
Exercise a correct solution, a different valid solution where one exists, and an
informative incorrect submission. These cases explain what the task measures.
Installed task providers can wrap existing solvers, simulators or databases while
the shared Episode records the interaction and terminal result.""",
    ),
    "episode-workflow": (
        "Operate an episode through its observable interface",
        """Call gym_tasks to discover families, then gym_open with a family, seed,
split and difficulty. Keep the returned episode_id. Read the initial prompt, tool
schemas, units, requested submission fields and action budget before choosing an
action. The task's observable interface defines the information available to this
attempt; reference implementations are separate baselines.

Use gym_action with {tool: name, arguments: {...}} to observe or act, and
{submit: {...}} for the final artifact. Choose the next action from the returned
state. Dependent actions within an episode are ordered; independent episodes can
run concurrently. A tool error may provide information for correcting a call.

Read the actual terminal result. passed and failed describe scored submissions;
error, truncated and cancelled retain unscored outcomes. Keep task failure and
missing measurement separate when computing policy statistics. Report the observed
resource use and outcome rather than assigning a success score from the narrative.

Use gym_close when abandoning an unfinished episode. A terminal episode is complete;
another attempt starts a new episode. Retain attempt counts when using retries.
For an external harness, the JSONL service exposes tasks, open, action and close
over the same operations. Use stdout for protocol messages and stderr for logs.""",
    ),
    "grouped-collection": (
        "Collect experience with an external batched policy",
        """A TaskSuite defines the finite task population and attempts per task.
Its sampler keeps each task's attempt group together across ranks, with explicit
epoch, seed and issuance order. PolicyVersion identifies the policy revision and
generation settings. Model inference and optimization remain in the learner.

The collector fills independent environment slots, asks the external policy for
actions over active observations, steps those slots and refills completed ones.
The policy can batch inference while each episode keeps its own ordered history.
Use the supplied policy-data channel for measured token statistics or other learner
annotations; do not manufacture probabilities from terminal rewards.

A collection resume retains passed, failed, truncated and error outcomes and
continues pending work; --retry-errors explicitly retries error outcomes. Cancelled
or interrupted live episodes restart in fresh environments; this is
not restoration of arbitrary simulator memory. Keep its restart history distinct
from a newly requested statistical attempt. Resume with the same suite, policy and
sampling configuration so the population has the same meaning.

Export completed trajectories with their task, group, attempt, policy, interactions
and terminal results. Preserve unscored outcomes when describing collection coverage.
Use the standalone collection example to connect a policy, then replace its action
function with the learner's own batched inference. A working trajectory pipeline is
the input to a learning experiment; it does not establish an improvement in policy.""",
    ),
    "policy-comparison": (
        "Compare policies under the intended interaction budget",
        """State which policy property changed and which behavior should improve.
Use the same task population, tools, initial information, verifier and resource
budget across policies. Record model, prompts, memory, search, decoding and tool
selection when they change the experimental condition. Counterbalance execution
order when shared services or changing external state could affect measurements.

Keep repeated attempts within tasks and related tasks within their independent
groups. First-attempt success, success in any of k attempts and the quality of a
selected answer are different outcomes. A system that can produce a successful
attempt may still lack a way to identify it. Include the selection mechanism when
the deployed policy must return one answer.

Retain scored failures, unscored runs, time and tool use. Compare both completed
outcomes and collection coverage; an apparent gain from omitting interrupted hard
tasks changes the population being measured. Resource measurements should include
the retries and search needed by the proposed policy.

Study concrete trajectories to explain a difference: information acquired, action
choices, intermediate state and the final artifact. Use changed tasks or controlled
component interventions to distinguish competing explanations. For learning studies,
keep evaluation tasks separate from the feedback used to update the policy. Frozen
suites support version comparisons; new suites measure transfer to new situations.""",
    ),
}


def get_guide(topic: str = "task-design") -> dict:
    """Read the guide for the current environment or policy question."""
    if topic not in GUIDES:
        raise ValueError(f"unknown guide; choose one of: {', '.join(GUIDES)}")
    title, content = GUIDES[topic]
    return {"topic": topic, "title": title, "content": content}


def design_task(capability: str, population: str, outcome: str) -> str:
    """Construct a task package around observable capability."""
    context = json.dumps({"capability": capability, "population": population, "outcome": outcome}, ensure_ascii=False)
    return ("Build an executable task for this research question: " + context + "\n"
            "Read gym://guides/task-design. Define initial information, actions, state changes and the "
            "submitted outcome. Reuse a domain engine or installed provider where it fits. Implement the "
            "shared Task interface and independent scoring, run a concrete episode, and show why valid "
            "alternative solutions receive the intended result. Return the package and example invocation.")


def collect_experience(suite: str, policy: str, budget: str) -> str:
    """Connect a policy to resumable grouped experience collection."""
    context = json.dumps({"suite": suite, "policy": policy, "budget": budget}, ensure_ascii=False)
    return ("Collect experience for this learner: " + context + "\n"
            "Read gym://guides/grouped-collection and discover the installed collection interface. "
            "Use the suite's actual specifications, explicit policy revision and attempt budget. Connect "
            "batched action generation, retain task/attempt identities and use resume for the same "
            "collection. Export trajectories and summarize completed outcomes, coverage and measured cost.")


def compare_policies(question: str, suite: str, policies: str) -> str:
    """Compare policy behavior through matched tasks and observed interactions."""
    context = json.dumps({"question": question, "suite": suite, "policies": policies}, ensure_ascii=False)
    return ("Investigate this policy difference: " + context + "\n"
            "Read gym://guides/policy-comparison. Hold task and interaction conditions fixed except for "
            "the intended change. Collect the planned attempts, retain unscored outcomes and compare "
            "quality at the relevant resource budget. Use a few trajectories to explain what changed, "
            "and distinguish the measured effect from a proposed mechanism. Return the run and export paths.")
