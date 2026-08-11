"""Imports and helpers shared by the assembled AI2-THOR executable.

``execute_plan.py`` places this file before ``aithor_connect.py``, generated
plan code, and ``end_thread.py``. Imports required by those files therefore
belong here even when they are not referenced directly below.
"""

import math
import os
import random
import re
import shutil
import subprocess
import threading
import time
from glob import glob
from typing import Tuple

import cv2
import numpy as np
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
from scipy.spatial import distance


def closest_nodes_collision_aware(
    goal_xyz,
    nodes,
    k,
    min_sep=0.50,
    avoid_pts=(),
    avoid_radius=0.55,
    linecheck=None,
):
    """Select nearby approach points while avoiding agents and obstacles."""
    ordered_indices = np.argsort(distance.cdist([goal_xyz], nodes)[0])
    selected = []

    def xz_distance(first, second):
        return math.hypot(first[0] - second[0], first[2] - second[2])

    for index in ordered_indices:
        candidate = nodes[index]
        if any(xz_distance(candidate, point) < avoid_radius for point in avoid_pts):
            continue
        if any(xz_distance(candidate, point) < min_sep for point in selected):
            continue
        if linecheck and not linecheck(candidate, goal_xyz):
            continue

        selected.append(candidate)
        if len(selected) == k:
            break

    return selected


# Retained for the movement-based GoToObject implementation and its fallbacks.
def near_same_goal(agent_ids, goal_xyz, tol=1.0):
    nearby_agents = []
    for agent_id in agent_ids:
        position = get_agent_metadata(agent_id)["position"]
        distance_to_goal = math.hypot(
            position["x"] - goal_xyz[0],
            position["z"] - goal_xyz[2],
        )
        if distance_to_goal <= tol:
            nearby_agents.append(agent_id)
    return nearby_agents


def sectorized_targets(goal_xyz, all_nodes, agent_positions):
    """Assign approach points from different angular sectors around a goal."""
    bearings = []
    for position in agent_positions:
        dx = position[0] - goal_xyz[0]
        dz = position[2] - goal_xyz[2]
        bearings.append((math.degrees(math.atan2(dx, dz)) % 360, position))

    remaining_nodes = list(all_nodes)
    targets = []
    for bearing, _ in sorted(bearings, key=lambda item: item[0]):
        candidates = []
        for node in remaining_nodes:
            node_bearing = math.degrees(
                math.atan2(node[0] - goal_xyz[0], node[2] - goal_xyz[2])
            ) % 360
            angular_distance = abs(((node_bearing - bearing + 180) % 360) - 180)
            if angular_distance <= 45:
                candidates.append(
                    (math.hypot(node[0] - goal_xyz[0], node[2] - goal_xyz[2]), node)
                )
        if candidates:
            _, target = min(candidates)
            targets.append(target)
            remaining_nodes.remove(target)

    targets.extend(remaining_nodes[: len(agent_positions) - len(targets)])
    return targets[: len(agent_positions)]


def _agent_ok(event, agent_id):
    metadata = event.events[agent_id].metadata
    return (
        metadata.get("lastActionSuccess", True),
        metadata.get("errorMessage", ""),
    )


def closest_nodes_unique(goal_xyz, nodes, k, min_sep=0.50):
    """Select up to ``k`` nearby nodes with minimum XZ separation."""
    ordered_indices = np.argsort(distance.cdist([goal_xyz], nodes)[0])
    selected = []
    for index in ordered_indices:
        candidate = nodes[index]
        if all(
            math.hypot(candidate[0] - point[0], candidate[2] - point[2]) >= min_sep
            for point in selected
        ):
            selected.append(candidate)
            if len(selected) == k:
                break
    return selected


def distance_pts(
    first: Tuple[float, float, float],
    second: Tuple[float, float, float],
) -> float:
    """Return planar XZ distance between two AI2-THOR positions."""
    return math.hypot(first[0] - second[0], first[2] - second[2])


def _first_frame_index(frame_paths):
    indices = []
    for path in frame_paths:
        match = re.search(r"img_(\d{5})\.png$", path)
        if match:
            indices.append(int(match.group(1)))
    return min(indices, default=0)


def generate_video(frame_rate=3, verbose=False, show_errors_only=True):
    """Create videos for recorded agent and top-view frame sequences."""
    base_directory = os.path.dirname(__file__)
    frame_directories = glob(os.path.join(base_directory, "agent_*"))
    frame_directories.append(os.path.join(base_directory, "top_view"))

    for frame_directory in frame_directories:
        if not os.path.isdir(frame_directory):
            continue

        frames = sorted(glob(os.path.join(frame_directory, "img_*.png")))
        if not frames:
            if verbose:
                print(f"Skipping {frame_directory}: no frames found.")
            continue

        view_name = os.path.basename(os.path.normpath(frame_directory))
        output_path = os.path.join(base_directory, f"video_{view_name}.mp4")
        command = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostats",
            "-start_number",
            str(_first_frame_index(frames)),
            "-framerate",
            str(frame_rate),
            "-i",
            os.path.join(frame_directory, "img_%05d.png"),
            "-c:v",
            "libopenh264",
            "-b:v",
            "2M",
            "-pix_fmt",
            "yuv420p",
            "-r",
            str(frame_rate),
            "-tag:v",
            "avc1",
            output_path,
        ]

        try:
            if show_errors_only:
                subprocess.run(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True,
                )
            else:
                subprocess.run(command, check=True)
        except FileNotFoundError:
            print("Unable to generate videos: ffmpeg is not installed.")
            return
        except subprocess.CalledProcessError as error:
            stderr = (error.stderr or b"").decode(errors="replace")
            details = "\n".join(stderr.strip().splitlines()[-10:])
            print(
                f"Video generation failed for {view_name} "
                f"(exit code {error.returncode}).\n{details}"
            )
