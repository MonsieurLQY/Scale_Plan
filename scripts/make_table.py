"""Build a comparison table from method evaluation summaries."""

from __future__ import annotations

import argparse
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

TABLE_HEADERS = [
    "Method",
    "Simple No. Tasks",
    "Simple TC",
    "Simple GCR",
    "Simple Exe",
    "Simple PT",
    "Complex No. Tasks",
    "Complex TC",
    "Complex GCR",
    "Complex Exe",
    "Complex PT",
    "Vague No. Tasks",
    "Vague TC",
    "Vague GCR",
    "Vague Exe",
    "Vague PT",
    "Overall Average TC",
    "Overall Average GCR",
    "Overall Average Exe",
    "Overall Average PT",
]

SUMMARY_FIELDS = {
    "Total Number of scenarios": "task_count",
    "Average Task Completion": "task_completion",
    "Average Goal Condition Recall": "goal_condition_recall",
    "Average Executability": "executability",
    "Average Planning Time (Successful Tasks)": "planning_time",
}


@dataclass(frozen=True)
class EvaluationSummary:
    task_count: int
    task_completion: float
    goal_condition_recall: float
    executability: float
    planning_time: float

    def table_values(self) -> list[int | float]:
        return [
            self.task_count,
            self.task_completion,
            self.goal_condition_recall,
            self.executability,
            self.planning_time,
        ]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs-root",
        type=Path,
        default=REPOSITORY_ROOT / "logs",
        help="Directory containing logs_<method> folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file (default: <logs-root>/evaluation_summary_table.txt).",
    )
    return parser.parse_args()


def read_summary(path: Path) -> EvaluationSummary:
    """Parse named metrics so summary line ordering does not matter."""
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip() in SUMMARY_FIELDS:
            values[SUMMARY_FIELDS[key.strip()]] = value.strip()

    missing = set(SUMMARY_FIELDS.values()) - values.keys()
    if missing:
        raise ValueError(f"missing fields: {', '.join(sorted(missing))}")

    return EvaluationSummary(
        task_count=int(values["task_count"]),
        task_completion=float(values["task_completion"]),
        goal_condition_recall=float(values["goal_condition_recall"]),
        executability=float(values["executability"]),
        planning_time=float(values["planning_time"]),
    )


def build_method_row(method: str, summaries: list[EvaluationSummary]) -> list:
    row = [method]
    for summary in summaries:
        row.extend(summary.table_values())

    total_tasks = sum(summary.task_count for summary in summaries)
    if total_tasks == 0:
        overall = [0.0, 0.0, 0.0, 0.0]
    else:
        metrics = (
            "task_completion",
            "goal_condition_recall",
            "executability",
            "planning_time",
        )
        overall = [
            sum(getattr(summary, metric) * summary.task_count for summary in summaries)
            / total_tasks
            for metric in metrics
        ]

    row.extend(f"{value:.2f}" for value in overall)
    return row


def format_table(rows: list[list], headers: list[str]) -> str:
    """Render a portable plain-text table without external dependencies."""
    string_rows = [[str(value) for value in row] for row in rows]
    widths = [
        max(len(header), *(len(row[index]) for row in string_rows))
        for index, header in enumerate(headers)
    ]

    def format_row(row: list[str]) -> str:
        return " | ".join(value.ljust(width) for value, width in zip(row, widths))

    separator = "-+-".join("-" * width for width in widths)
    return "\n".join((format_row(headers), separator, *(format_row(row) for row in string_rows)))


def main() -> int:
    args = parse_arguments()
    logs_root = args.logs_root.expanduser().resolve()
    rows = []

    for method in METHODS:
        method_directory = logs_root / f"logs_{method}"
        if not method_directory.is_dir():
            print(f"Skipping {method}: {method_directory} not found.", file=sys.stderr)
            continue

        summaries = []
        try:
            for category in TASK_CATEGORIES:
                summary_path = method_directory / category / "evaluation_summary.txt"
                summaries.append(read_summary(summary_path))
        except (OSError, ValueError) as error:
            print(f"Skipping {method}: {error}", file=sys.stderr)
            continue

        rows.append(build_method_row(method, summaries))

    if not rows:
        print(f"No complete evaluation summaries found under {logs_root}.", file=sys.stderr)
        return 1

    table = format_table(rows, TABLE_HEADERS)
    print(table)

    output_path = args.output or logs_root / "evaluation_summary_table.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        table + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
