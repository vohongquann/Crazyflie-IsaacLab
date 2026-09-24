# Drone_RL

An installable downstream Isaac Lab task package with a standard uv `src` layout.

No tasks are registered yet. Add task packages under `src/Drone_RL/tasks` and register them with Gymnasium.

## Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then create the project environment. The default
environment uses the Newton backend and does not install Isaac Sim:

```bash
uv sync
```

This project's `pyproject.toml` uses editable relative paths to an Isaac Lab source checkout. Update
`[tool.uv.sources]` if either directory moves.

The project forwards every optional extra declared by the Isaac Lab package used to create it, including physics,
rendering, visualizer, RL, teleoperation, and development features. Inspect `pyproject.toml` for the complete list and
pass each required extra to `uv run`:

```bash
# Standalone OV PhysX
uv run --extra ovphysx isaaclab random_agent --task <TASK_NAME> physics=ovphysx

# Isaac Sim with PhysX and RTX rendering
uv run --extra isaacsim isaaclab random_agent --task <TASK_NAME> physics=isaacsim_physx

# Isaac Sim with the Rerun visualizer
uv run --extra isaacsim --extra rerun isaaclab random_agent --task <TASK_NAME> --viz rerun
```

The `all` extra forwards Isaac Lab's `all` selection. The `ov` extra installs both the `ovphysx` and `ovrtx` runtimes;
you can also select either independently. Commit `pyproject.toml` and `uv.lock` so collaborators use the same
environment.

## Run the tasks

After registering a task, list it with:

```bash
uv run isaaclab list_envs --show_presets
```

## Project structure

Add task packages under `src/Drone_RL/tasks`. Import each task package from that directory so its Gymnasium
registrations load through the project's `isaaclab.tasks` entry point.

## Project assets

Keep project-owned USD files and related data under `src/Drone_RL/assets/data`. The asset module exposes a
stable path that works from an editable checkout and an installed wheel:

```python
from Drone_RL.assets import DRONE_RL_ASSETS_DIR

robot_usd_path = DRONE_RL_ASSETS_DIR / "robots" / "my_robot.usd"
```

Keep asset configuration in the `Drone_RL.assets` package and pass `str(robot_usd_path)` to configuration fields that
expect a string. USD files are configured for Git LFS by the project's `.gitattributes`; install Git LFS before adding
large binary assets.

## Development

Run the registration test and code-quality checks through the project environment:

```bash
uv run pytest
uv run pre-commit run --all-files
```

The test helpers under `source/isaaclab_tasks/test` in the Isaac Lab repository are not part of the installed
`isaaclab_tasks` package. Keep test fixtures in this project and use public Isaac Lab APIs. If you copy
`env_test_utils.py`, it becomes vendored code whose upstream changes you must track.

To configure VS Code or Cursor, run the `setup_python_env` task or invoke its command directly:

```bash
uv run isaaclab --editor
```

The setup command selects the active interpreter and writes a git-ignored `pyrightconfig.json`. The resulting
configuration inherits the project's checked-in Pyright settings and adds the Isaac Sim extensions, project `src` root,
and any Isaac Lab packages discovered in the active Python environment. This supports both Pylance in VS Code and
basedpyright in Cursor.

In VS Code, use Pylance and select the interpreter that ran the setup command. In Cursor, install the
[basedpyright extension](https://marketplace.visualstudio.com/items?itemName=detachhead.basedpyright) instead of
Pylance, select the same interpreter, and reload the window. Both language servers read `pyrightconfig.json`.

When using the `isaacsim` extra, include it while generating the editor configuration so the command can discover the
Isaac Sim installation:

```bash
uv run --extra isaacsim isaaclab --editor
```

For an Isaac Sim binaries installation that is not available in the project environment, provide its path explicitly:

```bash
# Linux
uv run isaaclab --editor --isaac_path <isaac-sim-path>

# Windows
uv run isaaclab --editor --isaac_path <isaac-sim-path>
```

## Isaac Sim UI extension

Add the project root to the Isaac Sim Extension Manager search paths, refresh, and enable the extension under
`Third Party`. Launch Isaac Sim through the project's `isaacsim` extra so the UI dependencies are available. Kit loads
`src/Drone_RL/ui_extension_example.py` through `config/extension.toml`.

## Troubleshooting

If Pylance or basedpyright cannot resolve modules, confirm that the selected interpreter matches the one used to run the
setup command, then reload the editor window. To add a missing extension or reduce indexing memory, edit the `extraPaths`
array in the root `pyrightconfig.json`; remove simulator extension directories that the project does not use.
