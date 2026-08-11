"""Run MAT2THOR experiments sequentially.

Each ``FloorPlan<number><suffix>.json`` file is processed in a deterministic
order. After successful experiment runs, plans are converted to executable
code with ``plantocode.py``.
"""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
from pathlib import Path


METHODS = (
    "Scale_Plan",
    "LLM_as_Planner",
    "LLM_PDDL",
    "lammap_only_PDDL",
    "lammap",
)
SCALE_PLAN_METHODS = {"Scale_Plan", "LLM_as_Planner", "LLM_PDDL"}
FLOOR_PLAN_PATTERN = re.compile(r"FloorPlan(?P<number>\d+)(?P<suffix>[A-Za-z]*)\.json")

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIRECTORY = REPOSITORY_ROOT / "data" / "MAT2THOR"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=METHODS,
        default="Scale_Plan",
        help="Planning method to evaluate (default: %(default)s).",
    )
    parser.add_argument(
        "--openai-api-key-file",
        default="api_key",
        help="Path to the OpenAI API key file, with or without .txt.",
    )
    parser.add_argument(
        "--gpt-version",
        default=None,
        help="Model name passed to the experiment script.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Run remaining floor plans after an experiment fails.",
    )
    parser.add_argument(
        "--skip-plan-to-code",
        action="store_true",
        help="Do not run plantocode.py after the experiments.",
    )
    return parser.parse_args()


def discover_floor_plans(dataset_directory: Path) -> list[tuple[int, str, Path]]:
    """Return valid dataset files sorted by floor-plan number and suffix."""
    floor_plans = []
    for path in dataset_directory.glob("*.json"):
        match = FLOOR_PLAN_PATTERN.fullmatch(path.name)
        if match:
            floor_plans.append(
                (int(match.group("number")), match.group("suffix"), path)
            )
    return sorted(floor_plans, key=lambda item: (item[0], item[1]))


def build_experiment_command(
    method: str,
    floor_plan: int,
    suffix: str,
    api_key_file: str,
    gpt_version: str | None,
) -> list[str]:
    """Build the planner command appropriate for the selected method."""
    script_name = "Scale_Plan_DAG.py" if method in SCALE_PLAN_METHODS else "lamma_p.py"
    command = [
        sys.executable,
        str(REPOSITORY_ROOT / "scripts" / script_name),
        "--floor-plan",
        str(floor_plan),
        "--file-suffix",
        suffix,
        "--method",
        method,
        "--openai-api-key-file",
        api_key_file,
    ]
    if gpt_version:
        command.extend(("--gpt-version", gpt_version))
    return command


def run_command(command: list[str]) -> int:
    print(f"Running: {shlex.join(command)}", flush=True)
    return subprocess.run(command, cwd=REPOSITORY_ROOT, check=False).returncode


def main() -> int:
    args = parse_arguments()

    if not DATASET_DIRECTORY.is_dir():
        print(f"Dataset directory not found: {DATASET_DIRECTORY}", file=sys.stderr)
        return 2

    floor_plans = discover_floor_plans(DATASET_DIRECTORY)
    if not floor_plans:
        print(f"No FloorPlan*.json files found in {DATASET_DIRECTORY}", file=sys.stderr)
        return 2

    failures: list[tuple[Path, int]] = []
    for floor_plan, suffix, path in floor_plans:
        command = build_experiment_command(
            args.method,
            floor_plan,
            suffix,
            args.openai_api_key_file,
            args.gpt_version,
        )
        return_code = run_command(command)
        if return_code:
            failures.append((path, return_code))
            if not args.continue_on_error:
                break

    # Plan-to-code should only consume a complete set of successful planner logs.
    if failures:
        print("\nFailed experiments:", file=sys.stderr)
        for path, return_code in failures:
            print(f"  {path.name}: exit code {return_code}", file=sys.stderr)
        return 1

    if not args.skip_plan_to_code:
        command = [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "plantocode.py"),
            "--logs-dir",
            str(REPOSITORY_ROOT / "logs" / f"logs_{args.method}"),
        ]
        return run_command(command)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
