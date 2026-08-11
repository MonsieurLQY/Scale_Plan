"""Run the complete experiment pipeline with parallel planning jobs.

For each selected method, this script runs all MAT2THOR planning experiments,
converts generated plans to code, executes the simulations, and evaluates the
results. A failed stage stops the pipeline for that method.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
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
SCRIPTS_DIRECTORY = REPOSITORY_ROOT / "scripts"
DATASET_DIRECTORY = REPOSITORY_ROOT / "data" / "MAT2THOR"


@dataclass(frozen=True)
class Experiment:
    floor_plan: int
    suffix: str
    source_file: Path


@dataclass(frozen=True)
class ExperimentResult:
    experiment: Experiment
    return_code: int


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=("all", *METHODS),
        default="all",
        help="Method to run, or 'all' to run every method (default: %(default)s).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=min(16, os.cpu_count() or 1),
        help="Maximum concurrent planning jobs (default: %(default)s).",
    )
    parser.add_argument(
        "--openai-api-key-file",
        default="api_key",
        help="Path to the OpenAI API key file, with or without .txt.",
    )
    parser.add_argument(
        "--gpt-version",
        default=None,
        help="Model name passed to planning and plan-to-code scripts.",
    )
    parser.add_argument(
        "--goto-mode",
        choices=("teleport", "navigation"),
        default="teleport",
        help="Use teleportation for evaluation or movement for videos.",
    )
    args = parser.parse_args()
    if args.max_workers < 1:
        parser.error("--max-workers must be at least 1")
    return args


def discover_experiments(dataset_directory: Path) -> list[Experiment]:
    """Discover valid dataset files in numeric floor-plan order."""
    experiments = []
    for source_file in dataset_directory.glob("*.json"):
        match = FLOOR_PLAN_PATTERN.fullmatch(source_file.name)
        if match:
            experiments.append(
                Experiment(
                    floor_plan=int(match.group("number")),
                    suffix=match.group("suffix"),
                    source_file=source_file,
                )
            )
    return sorted(
        experiments,
        key=lambda experiment: (experiment.floor_plan, experiment.suffix),
    )


def build_experiment_command(
    experiment: Experiment,
    method: str,
    api_key_file: str,
    gpt_version: str | None,
) -> list[str]:
    """Build the planner command appropriate for the selected method."""
    script_name = "Scale_Plan_DAG.py" if method in SCALE_PLAN_METHODS else "lamma_p.py"
    command = [
        sys.executable,
        str(SCRIPTS_DIRECTORY / script_name),
        "--floor-plan",
        str(experiment.floor_plan),
        "--file-suffix",
        experiment.suffix,
        "--method",
        method,
        "--openai-api-key-file",
        api_key_file,
    ]
    if gpt_version:
        command.extend(("--gpt-version", gpt_version))
    return command


def run_experiment(
    experiment: Experiment,
    method: str,
    api_key_file: str,
    gpt_version: str | None,
) -> ExperimentResult:
    command = build_experiment_command(
        experiment,
        method,
        api_key_file,
        gpt_version,
    )
    print(f"[{experiment.source_file.name}] Running: {shlex.join(command)}", flush=True)
    completed = subprocess.run(command, cwd=REPOSITORY_ROOT, check=False)
    return ExperimentResult(experiment, completed.returncode)


def run_planning_experiments(
    experiments: list[Experiment],
    method: str,
    max_workers: int,
    api_key_file: str,
    gpt_version: str | None,
) -> bool:
    """Run planning jobs concurrently and report all failed inputs."""
    worker_count = min(max_workers, len(experiments))
    print(
        f"Running {len(experiments)} {method} experiments with "
        f"{worker_count} workers.",
        flush=True,
    )

    results = []
    # Each worker blocks on a child process, so threads provide concurrency
    # without adding another layer of Python worker processes.
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                run_experiment,
                experiment,
                method,
                api_key_file,
                gpt_version,
            )
            for experiment in experiments
        ]
        for future in as_completed(futures):
            results.append(future.result())

    failures = sorted(
        (result for result in results if result.return_code != 0),
        key=lambda result: (
            result.experiment.floor_plan,
            result.experiment.suffix,
        ),
    )
    if not failures:
        return True

    print(f"\n{len(failures)} planning experiment(s) failed:", file=sys.stderr)
    for result in failures:
        print(
            f"  {result.experiment.source_file.name}: "
            f"exit code {result.return_code}",
            file=sys.stderr,
        )
    return False


def run_pipeline_stage(name: str, command: list[str]) -> bool:
    print(f"\n{name}: {shlex.join(command)}", flush=True)
    completed = subprocess.run(command, cwd=REPOSITORY_ROOT, check=False)
    if completed.returncode == 0:
        return True
    print(f"{name} failed with exit code {completed.returncode}.", file=sys.stderr)
    return False


def run_method_pipeline(
    method: str,
    experiments: list[Experiment],
    args: argparse.Namespace,
) -> bool:
    """Run planning, conversion, simulation, and evaluation for one method."""
    print(f"\n{'=' * 72}\nRunning pipeline for {method}\n{'=' * 72}", flush=True)
    if not run_planning_experiments(
        experiments,
        method,
        args.max_workers,
        args.openai_api_key_file,
        args.gpt_version,
    ):
        return False

    plan_to_code_command = [
        sys.executable,
        str(SCRIPTS_DIRECTORY / "plantocode.py"),
        "--logs-dir",
        str(REPOSITORY_ROOT / "logs" / f"logs_{method}"),
        "--openai-api-key-file",
        args.openai_api_key_file,
    ]
    if args.gpt_version:
        plan_to_code_command.extend(("--gpt-version", args.gpt_version))

    # ``all`` short-circuits below, preventing later stages from consuming
    # incomplete output produced by an earlier failed stage.
    stages = (
        ("Plan-to-code conversion", plan_to_code_command),
        (
            "Simulation",
            [
                sys.executable,
                str(SCRIPTS_DIRECTORY / "run_all_simulations_parallel.py"),
                "--method",
                method,
                "--goto-mode",
                args.goto_mode,
            ],
        ),
        (
            "Evaluation",
            [
                sys.executable,
                str(SCRIPTS_DIRECTORY / "evaluate_results.py"),
                "--method",
                method,
            ],
        ),
    )
    return all(run_pipeline_stage(name, command) for name, command in stages)


def main() -> int:
    args = parse_arguments()
    if not DATASET_DIRECTORY.is_dir():
        print(f"Dataset directory not found: {DATASET_DIRECTORY}", file=sys.stderr)
        return 2

    experiments = discover_experiments(DATASET_DIRECTORY)
    if not experiments:
        print(f"No FloorPlan*.json files found in {DATASET_DIRECTORY}", file=sys.stderr)
        return 2

    methods = METHODS if args.method == "all" else (args.method,)
    failed_methods = []
    for method in methods:
        if not run_method_pipeline(method, experiments, args):
            failed_methods.append(method)

    if failed_methods:
        print(
            f"\nFailed method pipelines: {', '.join(failed_methods)}",
            file=sys.stderr,
        )
        return 1

    print("\nAll requested method pipelines completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
