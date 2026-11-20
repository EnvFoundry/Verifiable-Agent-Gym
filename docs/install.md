# Installation and host integration

There are two installation routes. Choose one; installing a Skill alone does not install an executable environment.

## Python package and any MCP client

With Python 3.11+ available, clone or extract this repository and run:

```sh
python -m pip install ".[mcp]"
agent-gym connect --output agent-gym.mcp.json
```

Merge the generated `mcpServers.agent-gym` entry into your host's MCP configuration. Hosts using a different settings format can reuse its `command` and `args`. This resolves the interpreter at installation time instead of assuming that every OS provides a command named `python`. Regenerate the configuration if that interpreter moves.

The actual service command is:

```sh
python -m agent_gym mcp
```

For a data-backed task population, generate a configuration that loads it:

```sh
agent-gym --task-pack warehouse-sql=/absolute/path/tasks.jsonl connect --output warehouse.mcp.json
```

The generated command keeps the pack's absolute path. See [task packages](task-packages.md) for data profiles and [training](training.md) for Python batched rollouts.

The host starts a stdio subprocess. No HTTP address, port, browser, or model key is needed. To set the output directory or concurrent episode capacity, append `--output /path/to/runs --max-active 16`. The default storage location is `~/.agent-gym/runs`; `AGENT_GYM_RUNS` overrides it. Paths are arguments, not shell fragments.

For Research OS's plugin configuration format:

```sh
agent-gym connect --host research-os --output agent-gym.plugin.json
```

This file uses the installed Python environment with MCP support. It is a configuration generator, not a claim that a particular Research OS desktop release has already been tested. Install the companion Skill using the host's Skill import facility if desired.

## Native plugins

The repository/plugin ZIP includes Claude Code and Codex manifests, MCP launch configurations, the Python source, and one common Skill. Both plugin launchers require **uv on the host's PATH**. The PEP 723 entry script declares Python 3.11+ and the MCP SDK dependency; uv prepares and caches that environment on first use. Initial setup needs network access unless those dependencies are already cached.

### Claude Code

Add the repository as a marketplace and install its plugin:

```text
/plugin marketplace add EnvFoundry/Verifiable-Agent-Gym
/plugin install verifiable-agent-gym@verifiable-agent-gym
```

For local development, start Claude Code with `--plugin-dir /absolute/path/to/Verifiable-Agent-Gym`. The MCP configuration uses Claude's `${CLAUDE_PLUGIN_ROOT}` expansion. The repository must remain intact; copying only `.mcp.json` loses its source entry point.

### Codex

Add this repository's catalog and install the plugin:

```sh
codex plugin marketplace add EnvFoundry/Verifiable-Agent-Gym
codex plugin add verifiable-agent-gym@verifiable-agent-gym
```

The catalog is `.agents/plugins/marketplace.json`. The package supplies a portable root `plugin.json` and `mcp.json` with a stdio transport and plugin-relative working directory. A `.codex-plugin/plugin.json` overlay and `mcp.codex.json` retain compatibility with hosts using that layout. Neither Codex configuration assumes it expands Claude's root variable.

If your Codex version does not expose plugin installation, use the generated MCP configuration through its normal MCP settings and copy `skills/verifiable-agent-gym/` into its Skill directory. MCP integration does not require native plugin support.

### Other hosts

Use the package's generated command/arguments with any compatible MCP stdio client. A host can expose `gym_action` directly or translate each episode's `tools` list into native function calls. Its action must still pass through the Gym so observations and rewards are recorded consistently. No provider-specific API is embedded in the Gym.

The root portable manifest targets [Agent Plugins 1.0.0](https://agent-plugins.org/). These manifests are shipping formats, not automatic listings in an external vendor's curated store. Marketplace acceptance and host-specific UI installation remain separate from protocol compatibility.

## Node harnesses and installed guidance

The [TypeScript SDK](../sdk/typescript/README.md) supplies typed command operations, a persistent JSONL environment client and a helper for batched policies. Build it with `npm ci` and `npm run build` in `sdk/typescript`; it invokes the independently installed Python package. No model SDK or HTTP service is required.

Use `agent-gym capabilities` or `gym_capabilities` for the installed surface. Method resources under `gym://guides/` cover task design, interaction, collection and policy comparison. Collection tools return progress; rollout pages expose selected records without sending full traces by default. [Collection](collection.md) defines the exact operation and transport fields.

## Distribution

`python -m pip wheel . --no-deps --wheel-dir dist` builds the core wheel. `python scripts/package_plugin.py` creates a source/plugin ZIP and a Skill-only ZIP. Released GitHub source archives can also be installed directly. Nothing is published to PyPI under this name by these commands.

Use `python -m unittest discover -s tests -v` after installing the MCP extra to exercise real stdio discovery, concurrent episodes, verification, and subprocess cleanup. CI additionally installs the wheel and extracted plugin in new locations on each supported operating system.
