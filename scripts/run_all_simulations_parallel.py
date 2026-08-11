"""Execute generated AI2-THOR plans concurrently.

The script discovers experiment directories under ``logs/logs_<method>`` and runs
``execute_plan.py`` once per directory. Timed-out simulations are terminated
with their child processes and retried up to the configured number of attempts.
"""

from __future__ import annotations

import argparse
import os
import shlex
import signal
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
TASK_CATEGORIES = ("simple_tasks", "complex_tasks", "vague_tasks")
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXECUTE_PLAN_SCRIPT = REPOSITORY_ROOT / "scripts" / "execute_plan.py"


@dataclass(frozen=True)
class SimulationResult:
    directory: Path
    succeeded: bool
    message: str


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=METHODS,
        default="Scale_Plan",
        help="Planning method whose generated plans should be run.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Maximum number of concurrent simulations (default: %(default)s).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120,
        help="Timeout in seconds for each attempt (default: %(default)s).",
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=3,
        help="Maximum attempts per simulation (default: %(default)s).",
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
    if args.timeout <= 0:
        parser.error("--timeout must be greater than 0")
    if args.attempts < 1:
        parser.error("--attempts must be at least 1")
    return args


def discover_simulation_directories(log_directory: Path) -> list[Path]:
    """Find experiment directories in a stable category/name order."""
    directories = []
    for category in TASK_CATEGORIES:
        category_directory = log_directory / category
        if category_directory.is_dir():
            directories.extend(
                sorted(
                    (path for path in category_directory.iterdir() if path.is_dir()),
                    key=lambda path: path.name,
                )
            )
    return directories


def terminate_process_tree(process: subprocess.Popen[bytes]) -> None:
    """Terminate a simulation and the child processes in its process group."""
    if process.poll() is not None:
        return

    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
            return
        except (ProcessLookupError, subprocess.TimeoutExpired):
            pass

        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    else:
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def run_with_timeout(command: list[str], timeout: float) -> int | None:
    """Run a command, returning ``None`` when the command times out."""
    process = subprocess.Popen(
        command,
        cwd=REPOSITORY_ROOT,
        start_new_session=os.name == "posix",
        creationflags=(
            subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        ),
    )
    try:
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        terminate_process_tree(process)
        return None


def run_simulation(
    directory: Path,
    method: str,
    timeout: float,
    attempts: int,
    goto_mode: str,
) -> SimulationResult:
    """Run one generated plan until it succeeds or exhausts its attempts.

    A zero exit code alone is insufficient: ``execute_plan.py`` must also
    produce the expected ``results.txt`` artifact.
    """
    command = [
        sys.executable,
        str(EXECUTE_PLAN_SCRIPT),
        "--command",
        str(directory),
        "--method",
        method,
        "--goto-mode",
        goto_mode,
    ]

    for attempt in range(1, attempts + 1):
        print(
            f"[{directory.name}] attempt {attempt}/{attempts}: "
            f"{shlex.join(command)}",
            flush=True,
        )
        return_code = run_with_timeout(command, timeout)
        if return_code is None:
            print(f"[{directory.name}] timed out after {timeout:g} seconds", flush=True)
            continue
        if return_code != 0:
            print(f"[{directory.name}] exited with code {return_code}", flush=True)
            continue
        if (directory / "results.txt").is_file():
            return SimulationResult(directory, True, "completed")

        print(f"[{directory.name}] did not create results.txt", flush=True)

    return SimulationResult(directory, False, f"failed after {attempts} attempts")


def main() -> int:
    args = parse_arguments()
    log_directory = REPOSITORY_ROOT / "logs" / f"logs_{args.method}"
    if not log_directory.is_dir():
        print(f"Log directory not found: {log_directory}", file=sys.stderr)
        return 2

    directories = discover_simulation_directories(log_directory)
    if not directories:
        print(f"No simulation directories found under {log_directory}", file=sys.stderr)
        return 2

    worker_count = min(args.max_workers, len(directories))
    print(
        f"Running {len(directories)} simulations with {worker_count} workers.",
        flush=True,
    )

    results = []
    # Threads are sufficient because each worker waits on an external process;
    # they also avoid the serialization overhead of a process pool.
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                run_simulation,
                directory,
                args.method,
                args.timeout,
                args.attempts,
                args.goto_mode,
            ): directory
            for directory in directories
        }
        for future in as_completed(futures):
            directory = futures[future]
            try:
                result = future.result()
            except Exception as error:
                result = SimulationResult(directory, False, str(error))
            results.append(result)
            status = "completed" if result.succeeded else result.message
            print(f"[{directory.name}] {status}", flush=True)

    failures = sorted(
        (result for result in results if not result.succeeded),
        key=lambda result: str(result.directory),
    )
    if failures:
        print(f"\n{len(failures)} simulation(s) failed:", file=sys.stderr)
        for result in failures:
            print(f"  {result.directory}: {result.message}", file=sys.stderr)
        return 1

    print(f"All {len(results)} simulations completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
