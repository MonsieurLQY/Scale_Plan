"""Aggregate simulation metrics for one planning method."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


METHODS = (
    "Scale_Plan",
    "LLM_as_Planner",
    "LLM_PDDL",
    "lammap_only_PDDL",
    "lammap",
)
TASK_CATEGORIES = ("simple_tasks", "complex_tasks", "vague_tasks")
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

NUMBER_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
METRIC_PATTERNS = {
    "task_completion": re.compile(rf"Task Completion:\s*({NUMBER_PATTERN})"),
    "goal_condition_recall": re.compile(
        rf"Goal Condition Recall:\s*({NUMBER_PATTERN})"
    ),
    "executability": re.compile(rf"Executability:\s*({NUMBER_PATTERN})"),
    "planning_time": re.compile(rf"Planningtime:\s*({NUMBER_PATTERN})"),
}


@dataclass(frozen=True)
class Metrics:
    task_completion: float
    goal_condition_recall: float
    executability: float
    planning_time: float


@dataclass(frozen=True)
class EvaluationSummary:
    scenario_count: int
    result_count: int
    average_task_completion: float
    average_goal_condition_recall: float
    average_executability: float
    average_planning_time: float


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=METHODS,
        default="Scale_Plan",
        help="Method whose simulation results should be evaluated.",
    )
    parser.add_argument(
        "--logs-root",
        type=Path,
        default=REPOSITORY_ROOT / "logs",
        help="Directory containing logs_<method> folders.",
    )
    return parser.parse_args()


def parse_metrics(results_path: Path) -> Metrics:
    """Read the final occurrence of each metric from a simulation result."""
    content = results_path.read_text(encoding="utf-8")
    values = {}
    for name, pattern in METRIC_PATTERNS.items():
        matches = pattern.findall(content)
        if not matches:
            raise ValueError(f"missing {name.replace('_', ' ')}")
        values[name] = float(matches[-1])
    return Metrics(**values)


def _average(values: list[float], digits: int) -> float:
    return round(sum(values) / len(values), digits) if values else 0.0


def evaluate_category(category_directory: Path) -> EvaluationSummary:
    scenario_directories = sorted(
        (path for path in category_directory.iterdir() if path.is_dir()),
        key=lambda path: path.name,
    )
    metrics = []
    for scenario_directory in scenario_directories:
        results_path = scenario_directory / "results.txt"
        if not results_path.is_file():
            continue
        try:
            metrics.append(parse_metrics(results_path))
        except (OSError, ValueError) as error:
            print(f"Skipping malformed result {results_path}: {error}", file=sys.stderr)

    successful_planning_times = [
        result.planning_time for result in metrics if result.task_completion > 0
    ]
    return EvaluationSummary(
        scenario_count=len(scenario_directories),
        result_count=len(metrics),
        average_task_completion=_average(
            [result.task_completion for result in metrics], 2
        ),
        average_goal_condition_recall=_average(
            [result.goal_condition_recall for result in metrics], 2
        ),
        average_executability=_average(
            [result.executability for result in metrics], 2
        ),
        average_planning_time=_average(successful_planning_times, 1),
    )


def write_summary(path: Path, summary: EvaluationSummary) -> None:
    """Write the format consumed by ``make_table.py``."""
    content = (
        f"Total Number of scenarios: {summary.scenario_count}\n"
        f"Results found in scenarios: {summary.result_count}\n"
        f"Average Task Completion: {summary.average_task_completion}\n"
        f"Average Goal Condition Recall: {summary.average_goal_condition_recall}\n"
        f"Average Executability: {summary.average_executability}\n"
        "Average Planning Time (Successful Tasks): "
        f"{summary.average_planning_time}\n"
    )
    path.write_text(content, encoding="utf-8")


def print_summary(category_directory: Path, summary: EvaluationSummary) -> None:
    print(f"\n{category_directory.name}: {summary.scenario_count} scenarios")
    print(f"Results found: {summary.result_count}")
    print(f"Average Task Completion: {summary.average_task_completion}")
    print(f"Average Goal Condition Recall: {summary.average_goal_condition_recall}")
    print(f"Average Executability: {summary.average_executability}")
    print(
        "Average Planning Time (Successful Tasks): "
        f"{summary.average_planning_time}"
    )


def main() -> int:
    args = parse_arguments()
    method_directory = args.logs_root.expanduser().resolve() / f"logs_{args.method}"
    if not method_directory.is_dir():
        print(f"Log directory not found: {method_directory}", file=sys.stderr)
        return 2

    for category in TASK_CATEGORIES:
        category_directory = method_directory / category
        if not category_directory.is_dir():
            print(f"Task category not found: {category_directory}", file=sys.stderr)
            return 2

        summary = evaluate_category(category_directory)
        write_summary(category_directory / "evaluation_summary.txt", summary)
        print_summary(category_directory, summary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
