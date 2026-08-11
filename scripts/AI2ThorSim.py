"""Load object metadata from an AI2-THOR floor plan."""

from __future__ import annotations

import argparse
import copy
import os
import tempfile
from typing import Any


__all__ = ["get_ai2_thor_objects", "get_ai2_thor_objects_randomized"]

OBJECT_FIELDS = (
    "objectType",
    "mass",
    "visible",
    "isInteractable",
    "distance",
    "position",
    "rotation",
    "objectId",
)


def _controller_class():
    """Import AI2-THOR lazily so module inspection does not launch the runtime."""
    try:
        from ai2thor.controller import Controller
    except ImportError as error:
        raise RuntimeError(
            "AI2-THOR is required to load floor-plan object metadata"
        ) from error
    return Controller


def _cloud_rendering_platform():
    try:
        from ai2thor.platform import CloudRendering
    except ImportError as error:
        raise RuntimeError(
            "This project requires AI2-THOR CloudRendering support"
        ) from error
    return CloudRendering


def _stop_controller(controller) -> None:
    if controller is not None:
        controller.stop()


def _filtered_objects(objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {field: obj[field] for field in OBJECT_FIELDS if field in obj}
        for obj in objects
    ]


def get_ai2_thor_objects(floor_plan: int) -> list[dict[str, Any]]:
    """Return selected metadata fields from an unmodified floor plan."""
    controller = None
    try:
        controller = _controller_class()(
            agentMode="default",
            visibilityDistance=1.5,
            scene=f"FloorPlan{floor_plan}",
            platform=_cloud_rendering_platform(),
        )
        objects = controller.last_event.metadata.get("objects", [])
        return _filtered_objects(objects)
    finally:
        _stop_controller(controller)


def get_ai2_thor_objects_randomized(
    floor_plan: int,
    random_seed: int = 1,
) -> list[dict[str, Any]]:
    """Return full metadata after deterministic random object placement."""
    controller = None
    previous_temp_directory = os.environ.get("AI2THOR_TEMP_DIRECTORY")

    with tempfile.TemporaryDirectory(prefix="ai2thor-") as temp_directory:
        os.environ["AI2THOR_TEMP_DIRECTORY"] = temp_directory
        try:
            controller = _controller_class()(
                scene=f"FloorPlan{floor_plan}",
                width=300,
                height=300,
                renderDepthImage=False,
                renderInstanceSegmentation=False,
                renderClassImage=False,
                platform=_cloud_rendering_platform(),
            )
            controller.step(
                action="InitialRandomSpawn",
                randomSeed=random_seed,
                forceVisible=True,
                placeStationary=True,
                numPlacementAttempts=5,
            )
            objects = controller.last_event.metadata.get("objects", [])
            return copy.deepcopy(objects)
        finally:
            _stop_controller(controller)
            if previous_temp_directory is None:
                os.environ.pop("AI2THOR_TEMP_DIRECTORY", None)
            else:
                os.environ["AI2THOR_TEMP_DIRECTORY"] = previous_temp_directory


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("floor_plan", type=int, help="AI2-THOR floor-plan number")
    parser.add_argument("--randomized", action="store_true")
    parser.add_argument("--random-seed", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    if args.randomized:
        objects = get_ai2_thor_objects_randomized(args.floor_plan, args.random_seed)
    else:
        objects = get_ai2_thor_objects(args.floor_plan)
    print(f"Loaded {len(objects)} objects from FloorPlan{args.floor_plan}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
