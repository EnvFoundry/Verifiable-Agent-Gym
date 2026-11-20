"""Exercise shipped wheel, relocated plugin, and external task package."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import venv
import zipfile
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def run(*args, **kwargs):
    return subprocess.run([str(x) for x in args], check=True, **kwargs)


def main():
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    wheel = next((ROOT / "dist").glob(f"verifiable_agent_gym-{version}-*.whl"))
    plugin = ROOT / "dist" / f"verifiable-agent-gym-{version}-plugin.zip"
    with tempfile.TemporaryDirectory(prefix="gym installation ") as folder:
        workspace = Path(folder)
        venv.EnvBuilder(with_pip=True).create(workspace / "env")
        python = workspace / "env" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        run(python, "-m", "pip", "install", "--no-index", "--no-deps", wheel)
        run(python, "-m", "agent_gym", "run", "--count", "2", "--jobs", "3", "--output", workspace / "run", "--require-pass", cwd=workspace)
        with zipfile.ZipFile(plugin) as archive:
            archive.extractall(workspace / "plugin relocation")
        unpacked = workspace / "plugin relocation/verifiable-agent-gym"
        run(python, unpacked / "server.py", "run", "--count", "1", "--output", workspace / "plugin-run", "--require-pass", cwd=workspace)
        # Build the independent example with the build environment, then install it offline.
        run(sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", ROOT / "examples/calibration-package", "--wheel-dir", workspace)
        example = next(workspace.glob("agent_gym_calibration_example-*.whl"))
        run(python, "-m", "pip", "install", "--no-index", "--no-deps", example)
        command = json.dumps([str(python), str(unpacked / "examples/calibration-package/agent.py")])
        run(python, "-m", "agent_gym", "run", "--tasks", "sensor-calibration", "--agent", "command", "--command-json", command, "--count", "4", "--jobs", "2", "--output", workspace / "calibration", "--require-pass", cwd=workspace)
    print("Fresh wheel, relocated plugin, and independently installed task package passed.")


if __name__ == "__main__":
    main()
