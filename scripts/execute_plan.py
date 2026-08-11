"""Assemble and execute an AI2-THOR plan from an experiment directory."""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


METHODS = (
    "Scale_Plan",
    "LLM_as_Planner",
    "LLM_PDDL",
    "lammap_only_PDDL",
    "lammap",
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONNECTOR_DIRECTORY = REPOSITORY_ROOT / "scripts" / "aithor_connect"
SHARED_ASSIGNMENTS = {"robots", "action_queue", "task_over"}

RUNTIME_DEFAULTS = {
    "ground_truth": [],
    "Planningtime": 0.0,
}
REQUIRED_RUNTIME_VALUES = {"robots", "floor_plan"}
ASSIGNMENT_PATTERN = re.compile(
    r"^(robots|floor_plan|ground_truth|Planningtime)\s*=\s*(.+)$"
)


class GeneratedPlanTransformer(ast.NodeTransformer):
    """Protect shared runtime state and select the requested GoTo mode."""

    def __init__(self, goto_mode: str):
        self.goto_name = (
            "GoToObject_teleport" if goto_mode == "teleport" else "GoToObject"
        )

    def visit_Call(self, node: ast.Call):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id in {
            "GoToObject",
            "GoToObject_teleport",
        }:
            node.func.id = self.goto_name
        return node


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--command",
        type=Path,
        required=True,
        help="Experiment directory containing log.txt and code_plan_plantocode.py.",
    )
    parser.add_argument(
        "--method",
        choices=METHODS,
        default="Scale_Plan",
        help="Planning method associated with the experiment.",
    )
    parser.add_argument(
        "--goto-mode",
        choices=("teleport", "navigation"),
        default="teleport",
        help="Use teleportation for evaluation or movement for videos.",
    )
    return parser.parse_args()


def load_runtime_values(log_path: Path) -> dict[str, Any]:
    """Read literal runtime assignments from an experiment log."""
    values = dict(RUNTIME_DEFAULTS)
    for raw_line in log_path.read_text(encoding="utf-8").splitlines():
        match = ASSIGNMENT_PATTERN.fullmatch(raw_line.strip())
        if not match:
            continue
        name, expression = match.groups()
        try:
            values[name] = ast.literal_eval(expression)
        except (SyntaxError, ValueError) as error:
            raise ValueError(f"Invalid {name} assignment in {log_path}") from error

    missing = REQUIRED_RUNTIME_VALUES - values.keys()
    if missing:
        raise ValueError(
            f"Missing runtime values in {log_path}: {', '.join(sorted(missing))}"
        )
    if not isinstance(values["robots"], list) or not values["robots"]:
        raise ValueError(f"robots must be a non-empty list in {log_path}")
    if not isinstance(values["floor_plan"], (int, str)):
        raise ValueError(f"floor_plan must be an integer or string in {log_path}")
    return values


def transform_generated_plan(source: str, goto_mode: str) -> str:
    """Remove runtime-state redefinitions and rewrite GoTo calls."""
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise ValueError(f"Generated plan contains invalid Python: {error}") from error
    def is_shared_assignment(node: ast.stmt) -> bool:
        if isinstance(node, ast.Assign):
            return any(
                isinstance(target, ast.Name) and target.id in SHARED_ASSIGNMENTS
                for target in node.targets
            )
        return (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id in SHARED_ASSIGNMENTS
        )

    tree.body = [node for node in tree.body if not is_shared_assignment(node)]
    tree = GeneratedPlanTransformer(goto_mode).visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"


def _runtime_source(
    values: dict[str, Any],
    results_path: Path,
    goto_mode: str,
) -> str:
    assignments = {
        "robots": values["robots"],
        "floor_plan": values["floor_plan"],
        "ground_truth": values["ground_truth"],
        "Planningtime": values["Planningtime"],
        "results_path": str(results_path),
        "GOTO_MODE": goto_mode,
    }
    return "\n".join(f"{name} = {value!r}" for name, value in assignments.items())


def build_executable(experiment_directory: Path, goto_mode: str) -> Path:
    """Assemble, validate, and write one executable simulation plan."""
    experiment_directory = experiment_directory.expanduser().resolve()
    if not experiment_directory.is_dir():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_directory}")

    log_path = experiment_directory / "log.txt"
    generated_plan_path = experiment_directory / "code_plan_plantocode.py"
    for required_path in (log_path, generated_plan_path):
        if not required_path.is_file():
            raise FileNotFoundError(f"Required file not found: {required_path}")

    runtime_values = load_runtime_values(log_path)
    generated_plan = transform_generated_plan(
        generated_plan_path.read_text(encoding="utf-8"),
        goto_mode,
    )

    sections = (
        (CONNECTOR_DIRECTORY / "imports_aux_fn.py").read_text(encoding="utf-8"),
        _runtime_source(
            runtime_values,
            experiment_directory / "results.txt",
            goto_mode,
        ),
        (CONNECTOR_DIRECTORY / "aithor_connect.py").read_text(encoding="utf-8"),
        generated_plan,
        (CONNECTOR_DIRECTORY / "end_thread.py").read_text(encoding="utf-8"),
    )
    executable_source = "\n\n".join(section.rstrip() for section in sections) + "\n"
    compile(executable_source, "<assembled-ai2thor-plan>", "exec")

    executable_path = experiment_directory / "executable_plan.py"
    temporary_path = experiment_directory / "executable_plan.py.tmp"
    temporary_path.write_text(executable_source, encoding="utf-8")
    temporary_path.replace(executable_path)
    return executable_path


def main() -> int:
    args = parse_arguments()
    try:
        executable_path = build_executable(args.command, args.goto_mode)
    except (OSError, ValueError, SyntaxError) as error:
        print(f"Could not build executable plan: {error}", file=sys.stderr)
        return 1

    completed = subprocess.run(
        [sys.executable, str(executable_path)],
        cwd=REPOSITORY_ROOT,
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
