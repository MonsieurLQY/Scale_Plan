"""AI2-THOR controller state and actions for assembled executable plans.

This source is concatenated after ``imports_aux_fn.py`` by ``execute_plan.py``.
The generated plan chooses between movement-based ``GoToObject`` and
evaluation-oriented ``GoToObject_teleport`` explicitly.
"""

import os
import random
import secrets
from collections import defaultdict


GOTO_MODE = globals().get("GOTO_MODE", "navigation")

def _base_name(obj_or_pattern: str) -> str:
    return obj_or_pattern.split("|", 1)[0]

# Per-agent memory: last visited instance id for each base name
LAST_VISITED = defaultdict(dict)  # dict[agent_id][base_name] = full_object_id


SURFACE_PARENTS = {
    "CounterTop", "Table", "CoffeeTable", "DiningTable",
    "Desk", "KitchenIsland", "Shelf", "SideTable", "Dresser"
}
NEED_CHILD_SURFACE = {
    "Sink", "Toilet", "Bathtub", "Microwave", "Fridge", "Cabinet", "Drawer", "Oven"
}
def _base_name_of_id(obj_id: str) -> str:
    return obj_id.split("|", 1)[0]

def _get_objmeta_by_id(obj_id: str):
    event = safe_last_event()
    for obj in event.metadata["objects"]:
        if obj.get("objectId") == obj_id:
            return obj
    return None

def _is_place_surface(obj_meta):
    return bool(obj_meta and obj_meta.get("receptacle", False))

def _resolve_place_surface(target_recp_id: str) -> str:
    """Return a placeable receptacle id. Do NOT descend for surface parents like CounterTop."""
    # Child receptacle IDs can be used directly.
    if target_recp_id.count("|") >= 4:
        return target_recp_id

    parent_meta = _get_objmeta_by_id(target_recp_id)
    parent_base = _base_name_of_id(target_recp_id)

    # Surface parents: place directly on them (never descend).
    if parent_base in SURFACE_PARENTS and _is_place_surface(parent_meta):
        return target_recp_id

    # Need-child containers: prefer interior child receptacle if present.
    if parent_base in NEED_CHILD_SURFACE:
        ev = safe_last_event()
        objects = ev.metadata["objects"]
        prefix = target_recp_id + "|"

        # Prefer a child receptacle whose ID extends the parent ID.
        child_recp = [
            o for o in objects
            if isinstance(o.get("objectId"), str)
            and o["objectId"].startswith(prefix)
            and _is_place_surface(o)
        ]
        if child_recp:
            return max(child_recp, key=lambda o: o.get("receptacleCount", 0)).get("objectId")

        # Otherwise use a receptacle that references the parent.
        related = [
            o for o in objects
            if _is_place_surface(o)
            and o.get("parentReceptacles")
            and target_recp_id in o["parentReceptacles"]
        ]
        if related:
            return max(related, key=lambda o: o.get("receptacleCount", 0)).get("objectId")

    # Default: return parent id
    return target_recp_id

CTRL_LOCK = threading.Lock()

def safe_step(*args, **kwargs):
    """
    Thread-safe wrapper for c.step that accepts either:
      - keyword args: safe_step(action="...", agentId=...)
      - a single dict positional arg: safe_step({"action": "...", "agentId": ...})
    """
    if args:
        if len(args) == 1 and isinstance(args[0], dict):
            kwargs = {**args[0], **kwargs}
        else:
            raise TypeError("safe_step accepts only keyword args or a single dict positional arg.")
    with CTRL_LOCK:
        return c.step(**kwargs)

def safe_last_event():
    with CTRL_LOCK:
        return c.last_event
    
def get_agent_metadata(agent_id: int):
    ev = safe_last_event()
    # Request a multi-agent event when the cached event lacks this agent.
    need_refresh = not hasattr(ev, "events") or ev.events is None or len(ev.events) <= agent_id
    if need_refresh:
        ev = safe_step(action="GetReachablePositions")  # returns .events for all agents after Initialize
    return ev.events[agent_id].metadata["agent"]
    
def clear_agent_and_topview_dirs(base: str):
    """Remove frame directories created by an earlier simulation run."""
    for name in os.listdir(base):
        directory = os.path.join(base, name)
        is_frame_directory = name.startswith("agent_") or name == "top_view"
        if is_frame_directory and os.path.isdir(directory):
            shutil.rmtree(directory, ignore_errors=True)


total_exec = 0
success_exec = 0

log = ""



def dist_xz(first, second):
    return math.hypot(first["x"] - second["x"], first["z"] - second["z"])

def select_spaced_positions(candidates, k, min_sep=0.75, max_tries=20):
    """
    Greedy Poisson-disk style selection:
    - candidates: list of {"x","y","z"} reachable positions
    - k: number of agents to place
    - min_sep: required XZ separation between any two agents
    - max_tries: reshuffle attempts if we fail to get k positions
    Returns: list of k positions or [] if not possible.
    """
    if k <= 0:
        return []
    shuffled_candidates = candidates[:]
    for _ in range(max_tries):
        random.shuffle(shuffled_candidates)
        chosen = []
        for candidate in shuffled_candidates:
            if all(dist_xz(candidate, position) >= min_sep for position in chosen):
                chosen.append(candidate)
                if len(chosen) == k:
                    return chosen
        # Retry when the shuffled positions are too close together.
    return []  # not enough well-spaced positions

c = Controller(height=1000, width=1000, platform=CloudRendering)
c.reset("FloorPlan" + str(floor_plan))
no_robot = len(robots)

multi_agent_event = safe_step(
    action='Initialize',
    agentMode="default",
    snapGrid=True,
    gridSize=0.25,
    rotateStepDegrees=20,
    visibilityDistance=100,
    fieldOfView=90,
    agentCount=no_robot
)
event = safe_step(action="GetMapViewCameraProperties")
event = safe_step(action="AddThirdPartyCamera", **event.metadata["actionReturn"])

reachable_positions_ = safe_step(action="GetReachablePositions").metadata["actionReturn"]
reachable_positions = positions_tuple = [(p["x"], p["y"], p["z"]) for p in reachable_positions_]

MIN_AGENT_SEP = 0.5
spawn_positions = select_spaced_positions(reachable_positions_, no_robot, min_sep=MIN_AGENT_SEP)

if not spawn_positions:
    spawn_positions = select_spaced_positions(reachable_positions_, no_robot, min_sep=0.35)
    if not spawn_positions:
        raise RuntimeError(f"Could not find {no_robot} well-spaced spawn points. "
                           "Try reducing MIN_AGENT_SEP or agentCount.")

# Place agents at the selected reachable positions.
for i, init_pos in enumerate(spawn_positions):
    safe_step(dict(action="Teleport", position=init_pos, agentId=i))

# Add small yaw offsets to avoid symmetric starting poses.
for i in range(no_robot):
    yaw = secrets.choice([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330])
    safe_step(dict(action="RotateRight", degrees=yaw, agentId=i))

ev = safe_last_event()
objs = [obj["objectId"] for obj in ev.metadata["objects"]]
action_queue = []


task_over = False

recp_id = None

for i in range (no_robot):
    multi_agent_event = safe_step(action="LookDown", degrees=35, agentId=i)
verbose = True
SHOW_VIEWS = "NONE"  # options: "ALL", "TOP_ONLY", "NONE"

def exec_actions():
    global total_exec, success_exec, log
    base = os.path.dirname(__file__)
    clear_agent_and_topview_dirs(base)
    # Create per-agent frame directories.
    for i in range(no_robot):
        folder_name = "agent_" + str(i+1)
        folder_path = os.path.dirname(__file__) + "/" + folder_name
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    
    # Create the top-view frame directory.
    folder_name = "top_view"
    folder_path = os.path.dirname(__file__) + "/" + folder_name
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    img_counter = 0
    
    while not task_over:
        if len(action_queue) > 0:
            try:
                act = action_queue[0]
                if act['action'] == 'ObjectNavExpertAction':
                    multi_agent_event = safe_step(dict(action=act['action'], position=act['position'], agentId=act['agent_id']))
                    next_action = multi_agent_event.metadata['actionReturn']

                    if next_action != None:
                        multi_agent_event = safe_step(action=next_action, agentId=act['agent_id'], forceAction=True)
                
                elif act['action'] == 'MoveAhead':
                    multi_agent_event = safe_step(action="MoveAhead", agentId=act['agent_id'])
                    
                elif act['action'] == 'MoveBack':
                    multi_agent_event = safe_step(action="MoveBack", agentId=act['agent_id'])
                        
                elif act['action'] == 'RotateLeft':
                    multi_agent_event = safe_step(action="RotateLeft", degrees=act['degrees'], agentId=act['agent_id'])
                    
                elif act['action'] == 'RotateRight':
                    multi_agent_event = safe_step(action="RotateRight", degrees=act['degrees'], agentId=act['agent_id'])

                elif act['action'] == 'LookDown':
                    multi_agent_event = safe_step(action="LookDown", degrees=act['degrees'], agentId=act['agent_id'])
                
                elif act['action'] == 'LookUp':
                    multi_agent_event = safe_step(action="LookUp", degrees=act['degrees'], agentId=act['agent_id'])

                elif act['action'] == 'PickupObject':
                    multi_agent_event = safe_step(action="PickupObject", objectId=act['objectId'], agentId=act['agent_id'], forceAction=True)
                    if multi_agent_event.metadata['errorMessage'] != "":
                        print (multi_agent_event.metadata['errorMessage'])
                        log += f"\n[PickupObject] Failed to pick up object: {act['objectId']}. Error: {multi_agent_event.metadata['errorMessage']}\n"
                    else:
                        if 'act' in locals() and 'count' in act and act['count'] == False:
                            pass  # do not increment success_exec
                        else:
                            success_exec += 1
                            print("[PickupObject] success_exec incremented to", success_exec)
                        print("[PickupObject] Picked up object:", act['objectId'])
                        log += f"\n[PickupObject] Picked up object: {act['objectId']}\n"
 
                elif act['action'] == 'PutObject':
                    multi_agent_event = safe_step(action="PutObject", objectId=act['objectId'], agentId=act['agent_id'], forceAction=True)
                    err = multi_agent_event.metadata.get('errorMessage', '')
                    if "No valid positions to place object found" in err:
                        print("PutObject failed due to no valid positions. Attempting recovery.")
                        # Recover from a failed movement action.
                        safe_step(action="LookUp", degrees=60, agentId=act['agent_id'])
                        safe_step(action="RotateRight", degrees=20, agentId=act['agent_id'])
                        multi_agent_event = safe_step(action="PutObject", objectId=act['objectId'], agentId=act['agent_id'], forceAction=True)
                        err = multi_agent_event.metadata.get('errorMessage', '')
                    if multi_agent_event.metadata['errorMessage'] != "":
                        print (multi_agent_event.metadata['errorMessage'])
                        log += f"\n[PutObjectError] Failed to put object: {act['objectId']}. Error: {multi_agent_event.metadata['errorMessage']}\n"
                    else:
                        if 'act' in locals() and 'count' in act and act['count'] == False:
                            pass  # do not increment success_exec
                        else:
                            success_exec += 1
                            print("[PutObject] success_exec incremented to", success_exec)
                        print("[PutObject] Put object:", act['objectId'])
                        log += f"\n[PutObject] Put object: {act['objectId']}\n"
 
                elif act['action'] == 'ToggleObjectOn':
                    multi_agent_event = safe_step(action="ToggleObjectOn", objectId=act['objectId'], agentId=act['agent_id'], forceAction=True)
                    if multi_agent_event.metadata['errorMessage'] != "":
                        print (multi_agent_event.metadata['errorMessage'])
                        log += f"\n[SwitchOn] Failed to switch on object: {act['objectId']}. Error: {multi_agent_event.metadata['errorMessage']}\n"
                    else:
                        if 'act' in locals() and 'count' in act and act['count'] == False:
                            pass  # do not increment success_exec
                        else:
                            success_exec += 1
                            print("[SwitchOn] success_exec incremented to", success_exec)
                        print("[SwitchOn] Switched on object:", act['objectId'])
                        log += f"\n[SwitchOn] Switched on object: {act['objectId']}\n"
                
                elif act['action'] == 'ToggleObjectOff':
                    multi_agent_event = safe_step(action="ToggleObjectOff", objectId=act['objectId'], agentId=act['agent_id'], forceAction=True)
                    if multi_agent_event.metadata['errorMessage'] != "":
                        print (multi_agent_event.metadata['errorMessage'])
                        log += f"\n[SwitchOff] Failed to switch off object: {act['objectId']}. Error: {multi_agent_event.metadata['errorMessage']}\n"
                    else:
                        if 'act' in locals() and 'count' in act and act['count'] == False:
                            pass  # do not increment success_exec
                        else:
                            success_exec += 1
                            print("[SwitchOff] success_exec incremented to", success_exec)
                        print("[SwitchOff] Switched off object:", act['objectId'])
                        log += f"\n[SwitchOff] Switched off object: {act['objectId']}\n"
                    
                elif act['action'] == 'OpenObject':
                    multi_agent_event = safe_step(action="OpenObject", objectId=act['objectId'], agentId=act['agent_id'], forceAction=True)
                    if multi_agent_event.metadata['errorMessage'] != "":
                        print (multi_agent_event.metadata['errorMessage'])
                        log += f"\n[OpenObject] Failed to open object: {act['objectId']}. Error: {multi_agent_event.metadata['errorMessage']}\n"
                    else:
                        if 'act' in locals() and 'count' in act and act['count'] == False:
                            pass  # do not increment success_exec
                        else:
                            success_exec += 1
                            print("[OpenObject] success_exec incremented to", success_exec)
                        print("[OpenObject] Opened object:", act['objectId'])
                        log += f"\n[OpenObject] Opened object: {act['objectId']}\n"
                    
                elif act['action'] == 'CloseObject':
                    if "Blinds" not in act['objectId']:
                        multi_agent_event = safe_step(action="CloseObject", objectId=act['objectId'], agentId=act['agent_id'], forceAction=True)
                    else:
                        multi_agent_event = safe_step(action="Pass")

                    if multi_agent_event.metadata['errorMessage'] != "":
                        print (multi_agent_event.metadata['errorMessage'])
                        log += f"\n[CloseObject] Failed to close object: {act['objectId']}. Error: {multi_agent_event.metadata['errorMessage']}\n"
                    else:
                        if 'act' in locals() and 'count' in act and act['count'] == False:
                            print("Not incrementing success_exec for CloseObject as count is False")
                            pass  # do not increment success_exec
                        else:
                            success_exec += 1
                            print("[CloseObject] success_exec incremented to", success_exec)
                        print("[CloseObject] Closed object:", act['objectId'])
                        log += f"\n[CloseObject] Closed object: {act['objectId']}\n"
                        
                elif act['action'] == 'SliceObject':
                    multi_agent_event = safe_step(action="SliceObject", objectId=act['objectId'], agentId=act['agent_id'], forceAction=True)
                    if multi_agent_event.metadata['errorMessage'] != "":
                        print (multi_agent_event.metadata['errorMessage'])
                        log += f"\n[SliceObject] Failed to slice object: {act['objectId']}. Error: {multi_agent_event.metadata['errorMessage']}\n"
                    else:
                        if 'act' in locals() and 'count' in act and act['count'] == False:
                            pass  # do not increment success_exec
                        else:
                            success_exec += 1
                            print("[SliceObject] success_exec incremented to", success_exec)
                        print("[SliceObject] Sliced object:", act['objectId'])
                        log += f"\n[SliceObject] Sliced object: {act['objectId']}\n"
                        
                elif act['action'] == 'ThrowObject':
                    multi_agent_event = safe_step(action="ThrowObject", moveMagnitude=7, agentId=act['agent_id'], forceAction=True)
                    if multi_agent_event.metadata['errorMessage'] != "":
                        print (multi_agent_event.metadata['errorMessage'])
                        log += f"\n[ThrowObject] Failed to throw object from agent: {act['agent_id']}. Error: {multi_agent_event.metadata['errorMessage']}\n"
                    else:
                        if 'act' in locals() and 'count' in act and act['count'] == False:
                            pass  # do not increment success_exec
                        else:
                            success_exec += 1
                            print("[ThrowObject] success_exec incremented to", success_exec)
                        print("[ThrowObject] Threw object from agent:", act['agent_id'])
                        log += f"\n[ThrowObject] Threw object from agent: {act['agent_id']}\n"
                        
                elif act['action'] == 'BreakObject':
                    multi_agent_event = safe_step(action="BreakObject", objectId=act['objectId'], agentId=act['agent_id'], forceAction=True)
                    if multi_agent_event.metadata['errorMessage'] != "":
                        print (multi_agent_event.metadata['errorMessage'])
                        log += f"\n[BreakObject] Failed to break object: {act['objectId']}. Error: {multi_agent_event.metadata['errorMessage']}\n"
                    else:
                        if 'act' in locals() and 'count' in act and act['count'] == False:
                            pass  # do not increment success_exec
                        else:
                            success_exec += 1
                            print("[BreakObject] success_exec incremented to", success_exec)
                        print("[BreakObject] Broke object:", act['objectId'])
                        log += f"\n[BreakObject] Broke object: {act['objectId']}\n"
                # AI2-THOR interaction actions supported by the runtime.

                elif act['action'] == 'CleanObject':
                    multi_agent_event = safe_step(action="CleanObject", objectId=act['objectId'], agentId=act['agent_id'], forceAction=True)
                    if multi_agent_event.metadata['errorMessage'] != "":
                        print (multi_agent_event.metadata['errorMessage'])
                        log += f"\n[CleanObject] Failed to clean object: {act['objectId']}. Error: {multi_agent_event.metadata['errorMessage']}\n"
                    else:
                        if 'act' in locals() and 'count' in act and act['count'] == False:
                            pass  # do not increment success_exec
                        else:
                            success_exec += 1
                            print("[CleanObject] success_exec incremented to", success_exec)
                        print("[CleanObject] Cleaned object:", act['objectId'])
                        log += f"\n[CleanObject] Cleaned object: {act['objectId']}\n"

                elif act['action'] == 'DirtyObject':
                    multi_agent_event = safe_step(action="DirtyObject", objectId=act['objectId'], agentId=act['agent_id'], forceAction=True)
                    if multi_agent_event.metadata['errorMessage'] != "":
                        print (multi_agent_event.metadata['errorMessage'])
                        log += f"\n[DirtyObject] Failed to dirty object: {act['objectId']}. Error: {multi_agent_event.metadata['errorMessage']}\n"
                    else:
                        if 'act' in locals() and 'count' in act and act['count'] == False:
                            pass  # do not increment success_exec
                        else:
                            success_exec += 1
                            print("[DirtyObject] success_exec incremented to", success_exec)
                        print("[DirtyObject] Dirtied object:", act['objectId'])
                        log += f"\n[DirtyObject] Dirtied object: {act['objectId']}\n"


                elif act['action'] == 'DropHandObject':
                    multi_agent_event = safe_step(action="DropHandObject", agentId=act['agent_id'], forceAction=True)
                    if multi_agent_event.metadata['errorMessage'] != "":
                        print (multi_agent_event.metadata['errorMessage'])
                        log += f"\n[DropHandObject] Failed to drop held object from agent: {act['agent_id']}. Error: {multi_agent_event.metadata['errorMessage']}\n"
                    else:
                        if 'act' in locals() and 'count' in act and act['count'] == False:
                            pass  # do not increment success_exec
                        else:
                            success_exec += 1
                            print("[DropHandObject] success_exec incremented to", success_exec)
                        print("[DropHandObject] Dropped held object from agent:", act['agent_id'])
                        log += f"\n[DropHandObject] Dropped held object from agent: {act['agent_id']}\n"
                    
                elif act['action'] == 'PushObject':
                    multi_agent_event = safe_step(action="PushObject", objectId=act['objectId'], agentId=act['agent_id'], moveMagnitude=act['moveMagnitude'], forceAction=True)
                    if multi_agent_event.metadata['errorMessage'] != "":
                        print (multi_agent_event.metadata['errorMessage'])
                        log += f"\n[PushObject] Failed to push object: {act['objectId']}. Error: {multi_agent_event.metadata['errorMessage']}\n"
                    else:
                        if 'act' in locals() and 'count' in act and act['count'] == False:
                            pass  # do not increment success_exec
                        else:
                            success_exec += 1
                            print("[PushObject] success_exec incremented to", success_exec)
                        print("[PushObject] Pushed object:", act['objectId'])
                        log += f"\n[PushObject] Pushed object: {act['objectId']}\n"

                elif act['action'] == 'PullObject':
                    multi_agent_event = safe_step(action="PullObject", objectId=act['objectId'], agentId=act['agent_id'], moveMagnitude=act['moveMagnitude'], forceAction=True)
                    if multi_agent_event.metadata['errorMessage'] != "":
                        print (multi_agent_event.metadata['errorMessage'])
                        log += f"\n[PullObject] Failed to pull object: {act['objectId']}. Error: {multi_agent_event.metadata['errorMessage']}\n"
                    else:
                        if 'act' in locals() and 'count' in act and act['count'] == False:
                            pass  # do not increment success_exec
                        else:
                            success_exec += 1
                            print("[PullObject] success_exec incremented to", success_exec)
                        print("[PullObject] Pulled object:", act['objectId'])
                        log += f"\n[PullObject] Pulled object: {act['objectId']}\n"
                
                elif act['action'] == 'Done':
                    multi_agent_event = safe_step(action="Done")

                else:
                    print("Unknown action:", act['action'])
                    
                    
            except Exception as error:
                print(f"Action execution failed: {error}")
                log += f"\nAction execution failed: {error}\n"
                action_queue.pop(0)
                continue

            # --- display & save frames ---                        
            for i,e in enumerate(multi_agent_event.events):
                if SHOW_VIEWS == "ALL":
                    cv2.imshow('agent%s' % i, e.cv2img)
                f_name = os.path.dirname(__file__) + "/agent_" + str(i+1) + "/img_" + str(img_counter).zfill(5) + ".png"
                cv2.imwrite(f_name, e.cv2img)
            ev = safe_last_event()
            top_view_rgb = cv2.cvtColor(ev.events[0].third_party_camera_frames[-1], cv2.COLOR_BGR2RGB)
            if SHOW_VIEWS in ["ALL", "TOP_ONLY"]:
                cv2.imshow('Top View', top_view_rgb)
            f_name = os.path.dirname(__file__) + "/top_view/img_" + str(img_counter).zfill(5) + ".png"
            cv2.imwrite(f_name, top_view_rgb)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            
            img_counter += 1    
            action_queue.pop(0)
        else:
            time.sleep(0.01)
       
actions_thread = threading.Thread(target=exec_actions)
actions_thread.start()

def sidestep(agent_id, right=True):
    action_queue.append({'action': 'RotateRight' if right else 'RotateLeft', 'degrees': 90, 'agent_id': agent_id})
    action_queue.append({'action': 'MoveAhead', 'agent_id': agent_id})
    action_queue.append({'action': 'RotateLeft' if right else 'RotateRight', 'degrees': 90, 'agent_id': agent_id})

def _live_agent_positions(agent_ids):
    return [ (get_agent_metadata(a)["position"]["x"],
              get_agent_metadata(a)["position"]["y"],
              get_agent_metadata(a)["position"]["z"]) for a in agent_ids ]

def _ray_ok(cand_xyz, goal_xyz):
    # Check whether a direct movement segment is reachable.
    try:
        ev = safe_step(action="TestPathToPoint", position=dict(x=cand_xyz[0], y=cand_xyz[1], z=cand_xyz[2]),
                       target=dict(x=goal_xyz[0], y=goal_xyz[1], z=goal_xyz[2]))
        return bool(ev.metadata.get("actionReturn", True))
    except Exception:
        return True  # be permissive if unsupported


def _normalize_robots(selected_robots):
    if isinstance(selected_robots, dict):
        selected_robots = [selected_robots]
    elif not isinstance(selected_robots, list):
        selected_robots = [
            robot for robot in robots if robot["name"] == str(selected_robots)
        ]
    if not selected_robots:
        raise ValueError("At least one valid robot is required")
    return selected_robots


def _agent_ids(selected_robots):
    robot_indices = {robot["name"]: index for index, robot in enumerate(robots)}
    try:
        return [robot_indices[robot["name"]] for robot in selected_robots]
    except KeyError as error:
        raise ValueError(f"Unknown robot: {error.args[0]}") from error
        

def GoToObject(rob, dest_obj, count=True):
    """
    Move one or more robots to an object through navigable positions.
    - Robust object resolution (exact -> case-insensitive startswith; prefer non-zero center).
    - Safe agent_id lookup by robot name against global `robots` list (no brittle name parsing).
    - Per-agent progress tracking, replanning with wraparound + escape nudges on repeated stalls.
    - Planar (XZ) distance to goal; final shortest-angle yaw alignment for each arrived agent.
    - Records LAST_VISITED per agent; sets recp_id for key receptacles.
    """

    global robots, recp_id, reachable_positions, action_queue, LAST_VISITED, total_exec, success_exec

    if count:
        total_exec += 1  # for the GoToObject action itself

    rob = _normalize_robots(rob)
    no_agents = len(rob)
    agent_ids = _agent_ids(rob)

    # ---------- gather scene objects ----------
    ev = safe_last_event()
    scene_objs = ev.metadata.get("objects", [])
    objs = [obj.get("objectId") for obj in scene_objs]
    objs_center = [obj.get("axisAlignedBoundingBox", {}).get("center", {'x':0.0,'y':0.0,'z':0.0})
                   for obj in scene_objs]

    # ---------- resolve destination object id + center ----------
    dest_obj_id, dest_obj_center = None, None
    if isinstance(dest_obj, str) and "|" in dest_obj:
        # given as "id|x|y|z"
        parts = dest_obj.split("|")
        if len(parts) >= 4:
            dest_obj_id = parts[0]
            dest_obj_center = {'x': float(parts[1]), 'y': float(parts[2]), 'z': float(parts[3])}
    else:
        # 1) exact id match
        for idx, oid in enumerate(objs):
            if oid == dest_obj:
                dest_obj_id = oid
                dest_obj_center = objs_center[idx]
                break
        # 2) case-insensitive startswith, prefer non-zero center
        if dest_obj_id is None:
            low = str(dest_obj).lower()
            for idx, oid in enumerate(objs):
                if oid and oid.lower().startswith(low):
                    cand_center = objs_center[idx]
                    dest_obj_id = oid
                    dest_obj_center = cand_center
                    if cand_center != {'x': 0.0, 'y': 0.0, 'z': 0.0}:
                        break

    if dest_obj_id is None or dest_obj_center is None:
        print("[GoToObject] destination not found:", dest_obj)
        return

    if verbose:
        print("Going to", dest_obj_id, dest_obj_center)
    dest_obj_pos = [dest_obj_center['x'], dest_obj_center['y'], dest_obj_center['z']]

    # ---------- helpers ----------
    def dist_xz(ax, az, bx, bz):
        return math.hypot(ax - bx, az - bz)

    def shortest_yaw_delta(curr_yaw_deg, target_yaw_deg):
        # Normalize the angle to [-180, 180].
        return (target_yaw_deg - curr_yaw_deg + 540.0) % 360.0 - 180.0

    # ---------- per-agent progress bookkeeping ----------
    dist_goals = [10.0] * no_agents
    prev_dist_goals = [10.0] * no_agents
    cnt_since_update = [0] * no_agents
    MAX_REPLANS = 8
    replans = [0] * no_agents

    def select_approach_points():
        live_positions = _live_agent_positions(agent_ids)
        points = closest_nodes_collision_aware(
            dest_obj_pos,
            reachable_positions,
            no_agents,
            min_sep=0.50,
            avoid_pts=live_positions,
            avoid_radius=0.55,
            linecheck=_ray_ok,
        )
        if len(points) < no_agents:
            points = closest_nodes_unique(
                dest_obj_pos,
                reachable_positions,
                no_agents,
                min_sep=0.40,
            )
        if len(points) < no_agents:
            raise RuntimeError("Could not find approach points for all robots")
        return points

    crps = select_approach_points()
    goal_thresh = 0.25  # planar xz tolerance

    # ---------- main loop: run until ALL agents are within threshold ----------
    while any(d > goal_thresh for d in dist_goals):
        # Keep the candidate list aligned with the agent count.
        if len(crps) < no_agents:
            crps = select_approach_points()

        for i, agent_id in enumerate(agent_ids):
            # Agents at the goal remain stationary while others continue.
            if dist_goals[i] <= goal_thresh:
                continue

            # get agent pose
            md = get_agent_metadata(agent_id)
            rx = md["position"]["x"]
            rz = md["position"]["z"]

            # current CRP for this agent
            crp = crps[i]
            tx, ty, tz = crp[0], crp[1], crp[2]

            # update progress
            prev_dist_goals[i] = dist_goals[i]
            dist_goals[i] = dist_xz(rx, rz, tx, tz)
            dist_del = abs(dist_goals[i] - prev_dist_goals[i])

            if dist_del < 0.2:
                cnt_since_update[i] += 1
            else:
                cnt_since_update[i] = 0

            # advance or replan
            if cnt_since_update[i] < 8:
                action_queue.append({
                    'action': 'ObjectNavExpertAction',
                    'position': dict(x=tx, y=ty, z=tz),
                    'agent_id': agent_id
                })
            else:
                replans[i] += 1

                if replans[i] > MAX_REPLANS:
                    sidestep(agent_id, right=bool(random.getrandbits(1)))
                    # Reset replanning after repeated lack of progress.
                    replans[i] = 0

                crps = select_approach_points()
                cnt_since_update[i] = 0

        time.sleep(0.2)

    # ---------- final alignment (per agent) ----------
    for i, agent_id in enumerate(agent_ids):
        md = get_agent_metadata(agent_id)
        rx = md["position"]["x"]
        rz = md["position"]["z"]
        ryaw = md["rotation"]["y"]

        dx = dest_obj_pos[0] - rx
        dz = dest_obj_pos[2] - rz
        target_yaw = math.degrees(math.atan2(dx, dz))       # [-180, 180]
        rot_delta = shortest_yaw_delta(ryaw, target_yaw)

        if abs(rot_delta) > 1.0:
            action_queue.append({
                'action': 'RotateRight' if rot_delta > 0 else 'RotateLeft',
                'degrees': abs(rot_delta),
                'agent_id': agent_id
            })

        # bookkeeping for exact instance visited
        try:
            LAST_VISITED[agent_id][_base_name(dest_obj_id)] = dest_obj_id
        except Exception:
            pass
    if verbose:
        print("Reached:", dest_obj_id, "\n")
    if count:
        success_exec+= 1

    # Retain the resolved receptacle instance for later actions.
    base = _base_name(dest_obj_id) if 'dest_obj_id' in locals() else None
    if base in {"Cabinet", "Fridge", "CounterTop"}:
        recp_id = dest_obj_id
def GoToObject_teleport(
    rob,
    dest_obj,
    count=True,
    horizon=30,
    standing=True,
    max_tp_tries=30,
):
    """
    Teleport agent(s) to the closest reachable point near ``dest_obj``.

    Key behavior:
    - Resolves destination object robustly (exact id -> case-insensitive startswith; supports "id|x|y|z").
    - For each agent, repeatedly tries Teleport to nearest reachable candidates until success.
    - Sets rotation/horizon/standing directly in Teleport so the agent ends up facing the object.
    """

    global robots, recp_id, reachable_positions, action_queue, LAST_VISITED, total_exec, success_exec, log

    

    if count:
        total_exec += 1

    requested_robots = rob
    rob = _normalize_robots(rob)
    log += f"\n[GoToObject_teleport] robots={requested_robots}, dest_obj={dest_obj}\n"
    no_agents = len(rob)
    agent_ids = _agent_ids(rob)

    # ---------- gather scene objects ----------
    ev = safe_last_event()
    scene_objs = ev.metadata.get("objects", [])
    objs = [obj.get("objectId") for obj in scene_objs]
    objs_center = [
        obj.get("axisAlignedBoundingBox", {}).get("center", {"x": 0.0, "y": 0.0, "z": 0.0})
        for obj in scene_objs
    ]

    # ---------- resolve destination object id + center ----------
    dest_obj_id, dest_obj_center = None, None
    if isinstance(dest_obj, str) and "|" in dest_obj:
        parts = dest_obj.split("|")
        if len(parts) >= 4:
            dest_obj_id = parts[0]
            dest_obj_center = {"x": float(parts[1]), "y": float(parts[2]), "z": float(parts[3])}
    else:
        # 1) exact id match
        for idx, oid in enumerate(objs):
            if oid == dest_obj:
                dest_obj_id = oid
                dest_obj_center = objs_center[idx]
                break
        # 2) case-insensitive startswith, prefer non-zero center
        if dest_obj_id is None:
            low = str(dest_obj).lower()
            for idx, oid in enumerate(objs):
                if oid and oid.lower().startswith(low):
                    cand_center = objs_center[idx]
                    dest_obj_id = oid
                    dest_obj_center = cand_center
                    if cand_center != {"x": 0.0, "y": 0.0, "z": 0.0}:
                        break

    if dest_obj_id is None or dest_obj_center is None:
        print("[GoToObject_teleport] destination not found:", dest_obj)
        log += f"[GoToObject_teleport] destination not found: {dest_obj}\n"
        return

    dest_obj_pos = [dest_obj_center["x"], dest_obj_center["y"], dest_obj_center["z"]]

    if verbose:
        print("[GoToObject_teleport] Teleporting to:", dest_obj_id, dest_obj_center)

    # ---------- helpers ----------
    def dist_xz_tuple(p, goal_xyz):
        return math.hypot(p[0] - goal_xyz[0], p[2] - goal_xyz[2])

    def yaw_face(from_x, from_z, to_x, to_z):
        # AI2-THOR convention: rotation.y in degrees, 0 is +Z, 90 is +X, 180 is -Z, 270 is -X
        dx = to_x - from_x
        dz = to_z - from_z
        return (math.degrees(math.atan2(dx, dz)) + 360.0) % 360.0

    def _nearest_candidates(goal_xyz, avoid_xyz=None, k=50):
        """
        Return a list of reachable_positions sorted by distance to goal.
        Optionally soft-avoid a set of avoid points (other agents) by inflating distance.
        """
        avoid_xyz = avoid_xyz or []
        cand = []
        for p in reachable_positions:
            d = dist_xz_tuple(p, goal_xyz)
            # Penalize positions near other agents.
            for ax, ay, az in avoid_xyz:
                if math.hypot(p[0] - ax, p[2] - az) < 0.55:
                    d += 5.0
            cand.append((d, p))
        cand.sort(key=lambda t: t[0])
        return [p for _, p in cand[:k]]

    def _teleport_try(agent_id, p):
        tx, ty, tz = p[0], p[1], p[2]
        target_yaw = yaw_face(tx, tz, dest_obj_pos[0], dest_obj_pos[2])

        ev2 = safe_step(
            action="Teleport",
            position=dict(x=tx, y=ty, z=tz),
            rotation=dict(x=0.0, y=target_yaw, z=0.0),
            horizon=horizon,
            standing=standing,
            agentId=agent_id,
            forceAction=True
        )

        ok = bool(ev2.metadata.get("lastActionSuccess", True))
        if not ok and verbose:
            print(f"[GoToObject_teleport] Teleport failed for agent {agent_id}:",
                  ev2.metadata.get("errorMessage", "(no errorMessage)"),
                  " target=", (tx, ty, tz))
            log += f"[GoToObject_teleport] Teleport failed for agent {agent_id}: {ev2.metadata.get('errorMessage', '(no errorMessage)')} target= {(tx, ty, tz)}\n"
        return ok

    # ---------- choose initial CRPs (collision-aware if available) ----------
    crps = None
    try:
        other_now = _live_agent_positions(agent_ids)
        crps = closest_nodes_collision_aware(
            dest_obj_pos,
            reachable_positions,
            no_agents,
            min_sep=0.50,
            avoid_pts=other_now,
            avoid_radius=0.55,
            linecheck=_ray_ok
        )
    except Exception:
        crps = None

    if not crps or len(crps) < no_agents:
        # Fall back to the nearest separated reachable positions.
        other_now = _live_agent_positions(agent_ids)
        nearest = _nearest_candidates(dest_obj_pos, avoid_xyz=other_now, k=max(50, no_agents * 10))
        if not nearest:
            print("[GoToObject_teleport] No reachable positions available.")
            log += "[GoToObject_teleport] No reachable positions available.\n"
            return
        crps = [nearest[min(i, len(nearest) - 1)] for i in range(no_agents)]

    # ---------- teleport each agent (retry if failure) ----------
    any_failed = False
    for i, agent_id in enumerate(agent_ids):
        # Try the selected position before other nearby candidates.
        other_now = _live_agent_positions([a for a in agent_ids if a != agent_id])
        candidates = [crps[i]] + _nearest_candidates(dest_obj_pos, avoid_xyz=other_now, k=60)

        success = False
        tries = 0
        for p in candidates:
            if tries >= max_tp_tries:
                break
            tries += 1
            if _teleport_try(agent_id, p):
                success = True
                break

        if not success:
            any_failed = True
            print(f"[GoToObject_teleport] Could not teleport agent {agent_id} near {dest_obj_id} after {tries} tries.")
            log += f"[GoToObject_teleport] Could not teleport agent {agent_id} near {dest_obj_id} after {tries} tries.\n"

        # Record the resolved instance only after a successful teleport.
        if success:
            try:
                LAST_VISITED[agent_id][_base_name(dest_obj_id)] = dest_obj_id
            except Exception:
                pass

    if verbose:
        print("[GoToObject_teleport] Done:", dest_obj_id, "failed_any=", any_failed, "\n")
        

    if count and not any_failed:
        success_exec += 1
        print("[GoToObject_teleport] success_exec incremented to", success_exec)
        log += f"[GoToObject_teleport] Teleported agent {agent_id} near {dest_obj_id}\n"

    # Retain the resolved receptacle instance for later actions.
    base = _base_name(dest_obj_id) if "dest_obj_id" in locals() else None
    if base in {"Cabinet", "Fridge", "CounterTop"}:
        recp_id = dest_obj_id
    time.sleep(1.0)  # brief pause after teleport(s)


def GoToObject_for_mode(*args, **kwargs):
    """Use the navigation mode selected by ``execute_plan.py``."""
    if GOTO_MODE == "teleport":
        return GoToObject_teleport(*args, **kwargs)
    return GoToObject(*args, **kwargs)


def PutObject(robot, put_obj, recp, count=True):
    global robots, total_exec, success_exec, log

    
    if count:
        total_exec += 1  # for the PutObject action itself
    if isinstance(robot, str):
        for r in robots:
            if r['name'] == robot:
                robot = r
                break

    log += f"\n[PutObject] robot={robot['name']}, put_obj={put_obj}, recp={recp}"    
    put_obj_base = _base_name(put_obj)
    recp_base = _base_name(recp)
    for ia, r in enumerate(robots):
        if r['name'] == robot['name']:
            agent_id = ia
            break
    
    if verbose:
        print("Putting Object:", put_obj_base, "at:", recp_base)

    # Resolve the receptacle object.
    ev = safe_last_event()
    objs = [obj["objectId"] for obj in ev.metadata["objects"]]
    recp_id = None
    recp_center = None
    for idx, obj in enumerate(objs):
        # Match explicit instance IDs exactly.
        if '|' in recp:
            if obj == recp:
                recp_id = obj
                recp_center = ev.metadata["objects"][idx]["axisAlignedBoundingBox"]["center"]
                if recp_center != {'x': 0.0, 'y': 0.0, 'z': 0.0}:
                    break
        else:
            # Otherwise match by base object name.
            if _base_name_of_id(obj) == recp_base:
                recp_id = obj
                recp_center = ev.metadata["objects"][idx]["axisAlignedBoundingBox"]["center"]
                if recp_center != {'x': 0.0, 'y': 0.0, 'z': 0.0}:
                    break

    # --- proximity check (XZ only) before navigating ---
    md = get_agent_metadata(agent_id)
    rx, rz = md["position"]["x"], md["position"]["z"]
    ox, oz = recp_center['x'], recp_center['z']

    # Maximum XZ distance at which navigation is unnecessary.
    REACH_THRESH = 2.0  # meters/units in XZ

    dist_xz = math.hypot(ox - rx, oz - rz)

    if dist_xz > REACH_THRESH:
        log += f"[PutObject] Current dist to object (XZ): {dist_xz} > thresh {REACH_THRESH}; Action: PutObject Failed.\n"
        if verbose:
            print("Current dist to object (XZ):", dist_xz, "> thresh", REACH_THRESH, "; Action: PutObject Failed.\n")
            return
        
    # Verify that the agent holds the intended object.
    ev = safe_last_event()
    inv = ev.events[agent_id].metadata.get("inventoryObjects", [])
    if not any(re.match(put_obj_base.lower(), o["objectId"].lower()) for o in inv):
        log += f"\n[PutObject] Agent {agent_id} isn’t holding {put_obj_base}; skipping PutObject.\n"
        print(f"[PutObject] Agent {agent_id} isn’t holding {put_obj_base}; skipping PutObject.")
        return

    # 2) Resolve receptacle instance id
    target_recp_id = None

    # Prefer an explicitly provided instance ID.
    if '|' in recp:
        target_recp_id = recp

    # Otherwise reuse the last visited instance of the base object.
    if target_recp_id is None:
        target_recp_id = LAST_VISITED[agent_id].get(recp_base)

    # Fall back to the nearest matching receptacle.
    if target_recp_id is None:
        objs = [obj["objectId"] for obj in ev.metadata["objects"]]
        objs_dists = [obj["distance"] for obj in ev.metadata["objects"]]
        dist_min = float("inf")
        for idx, obj in enumerate(objs):
            if _base_name_of_id(obj) == recp_base:
                d = objs_dists[idx]
                if d < dist_min:
                    dist_min = d
                    target_recp_id = obj
        if target_recp_id is None:
            log += f"[PutObject] No receptacle found for base '{recp_base}'.\n"
            print(f"[PutObject] No receptacle found for base '{recp_base}'.")
            return

    # Resolve the actual placeable surface, such as a SinkBasin.
    target_recp_id = _resolve_place_surface(target_recp_id)
    if _base_name_of_id(target_recp_id) != recp_base and recp_base in SURFACE_PARENTS:
        # Surface receptacles remain the placement target.
        target_recp_id = LAST_VISITED[agent_id].get(recp_base, target_recp_id)

    # 4) Navigate and place
    # GoToObject(robot, target_recp_id)
    # time.sleep(1)

    # Adjust the camera pose to improve raycast placement.
    action_queue.append({'action': 'LookDown', 'degrees': 10, 'agent_id': agent_id})
    time.sleep(0.1)
    
    if verbose:
        print("Put:", put_obj_base, "at", target_recp_id, "\n")

    action_queue.append({'action': 'PutObject', 'objectId': target_recp_id, 'agent_id': agent_id, 'count': count})
    time.sleep(1)



def collapsed_actions(robot, sw_obj, action):
    global robots, total_exec, success_exec, log
    total_exec += 1  # for the action itself
    if isinstance(robot, str):
        for r in robots:
            if r['name'] == robot:
                robot = r
                break
    sw_obj = sw_obj.split('|')[0]
    for ia, r in enumerate(robots):
        if r['name'] == robot['name']:
            agent_id = ia
            break
    ev = safe_last_event()
    # --- find a concrete instance and its center ---
    objs = [obj["objectId"] for obj in ev.metadata["objects"]]
    objs_center = [obj["axisAlignedBoundingBox"]["center"] for obj in ev.metadata["objects"]]
    sw_obj_id, sw_obj_center = None, None

    for idx, obj in enumerate(objs):
        if re.match(sw_obj.lower(), obj.lower()):
            sw_obj_id = obj
            sw_obj_center = objs_center[idx]
            if sw_obj_center != {'x': 0.0, 'y': 0.0, 'z': 0.0}:
                break

    if not sw_obj_id or not sw_obj_center:
        print(f"[{action}] Couldn’t find instance of {sw_obj}.")
        log += f"\n[{action}] Couldn’t find instance of {sw_obj}.\n"
        return
    # --- proximity check (XZ only) before navigating ---
    md = get_agent_metadata(agent_id)
    rx, rz = md["position"]["x"], md["position"]["z"]
    ox, oz = sw_obj_center['x'], sw_obj_center['z']

    # Maximum XZ distance at which navigation is unnecessary.
    REACH_THRESH = 2.0  # meters/units in XZ

    dist_xz = math.hypot(ox - rx, oz - rz)


    if dist_xz > REACH_THRESH:
        if verbose:
            print("Current dist to object (XZ):", dist_xz, "> thresh", REACH_THRESH, "; Failed Plan.")
        log += f"\n[{action}] Current dist to object (XZ): {dist_xz} > thresh {REACH_THRESH}; Failed Plan.\n"
        return
    time.sleep(1)
    if action == "PickupObject":
        action_queue.append({'action':'PickupObject', 'objectId':sw_obj_id, 'agent_id':agent_id})
    elif action == "SwitchOn":
        action_queue.append({'action':'ToggleObjectOn', 'objectId':sw_obj_id, 'agent_id':agent_id})
    elif action == "SwitchOff":
        action_queue.append({'action':'ToggleObjectOff', 'objectId':sw_obj_id, 'agent_id':agent_id})
    elif action == "OpenObject":
        action_queue.append({'action':'OpenObject', 'objectId':sw_obj_id, 'agent_id':agent_id})
    elif action == "CloseObject":
        action_queue.append({'action':'CloseObject', 'objectId':sw_obj_id, 'agent_id':agent_id})
    elif action == "BreakObject":
        action_queue.append({'action':'BreakObject', 'objectId':sw_obj_id, 'agent_id':agent_id})
    elif action == "SliceObject":
        action_queue.append({'action':'SliceObject', 'objectId':sw_obj_id, 'agent_id':agent_id})
    elif action == "ThrowObject":
        action_queue.append({'action':'ThrowObject', 'objectId':sw_obj_id, 'agent_id':agent_id})
    elif action == "CleanObject":
        action_queue.append({'action':'CleanObject', 'objectId':sw_obj_id, 'agent_id':agent_id})
    elif action == "WashObject":
        sink_obj_id, sink_obj_center = None, None
        for idx, obj in enumerate(objs):
            if re.match("sink", obj.lower()):
                sink_obj_id = obj
                sink_obj_center = objs_center[idx]
                if sink_obj_center != {'x': 0.0, 'y': 0.0, 'z': 0.0}:
                    break
        faucet_obj_id, faucet_obj_center = None, None
        for idx, obj in enumerate(objs):
            if re.match("faucet", obj.lower()):
                faucet_obj_id = obj
                faucet_obj_center = objs_center[idx]
                if faucet_obj_center != {'x': 0.0, 'y': 0.0, 'z': 0.0}:
                    break
        
        print("Washing", sw_obj, "at sink", sink_obj_id, "with faucet", faucet_obj_id)
        log += f"\n[WashObject] robot={robot['name']}, sw_obj={sw_obj}, sink={sink_obj_id}, faucet={faucet_obj_id}\n"
        GoToObject_for_mode(robot, sink_obj_id, count=False)
        time.sleep(1)
        PutObject(robot, sw_obj_id, sink_obj_id, count = False)
        time.sleep(1)
        print("Switching on faucet", faucet_obj_id)
        log += f"[WashObject] Switching on faucet {faucet_obj_id}\n"
        action_queue.append({'action':'ToggleObjectOn', 'objectId':faucet_obj_id, 'agent_id':agent_id, 'count': False})
        time.sleep(1)
        print("Waiting for wash...")
        time.sleep(1)
        print("Switching off faucet", faucet_obj_id)
        log += f"[WashObject] Switching off faucet {faucet_obj_id}\n"
        action_queue.append({'action':'ToggleObjectOff', 'objectId':faucet_obj_id, 'agent_id':agent_id, 'count': False})
        time.sleep(1)
        print("Picking up", sw_obj)
        log += f"[WashObject] Picking up {sw_obj_id} after wash\n"
        action_queue.append({'action':'PickupObject', 'objectId':sw_obj_id, 'agent_id':agent_id, 'count': False})
        time.sleep(1)

        objects = ev.metadata.get("objects", [])
        if objects[objs.index(sw_obj_id)]['isDirty'] == True:
            action_queue.append({'action':'CleanObject', 'objectId':sw_obj_id, 'agent_id':agent_id, 'count': False})

        success_exec+=1
        print("[WashObject] success_exec incremented to", success_exec)
        objects[objs.index(sw_obj_id)]['isDirty'] = False
        log += f"[WashObject] Washed: {sw_obj}\n"
        if verbose:
            print("Washed:", sw_obj, "\n")
    elif action == "DirtyObject":
        action_queue.append({'action':'DirtyObject', 'objectId':sw_obj_id, 'agent_id':agent_id})
    elif action == "DropHandObject":
        action_queue.append({'action':'DropHandObject', 'agent_id':agent_id})
    elif action == "PushObject":
        action_queue.append({'action':'PushObject', 'objectId':sw_obj_id, 'agent_id':agent_id, 'moveMagnitude':150.0})
    elif action == "PullObject":
        action_queue.append({'action':'PullObject', 'objectId':sw_obj_id, 'agent_id':agent_id, 'moveMagnitude':150.0})
    time.sleep(1)


def _run_named_action(robot, object_name, action, description):
    """Log and submit one of AI2-THOR's single-object actions."""
    global log

    robot_name = robot if isinstance(robot, str) else robot["name"]
    log += f"\n[{action}] robot={robot_name}, sw_obj={object_name}"
    if verbose:
        print(f"[{action}] {description}:", object_name)
    collapsed_actions(robot, object_name, action)


def PickupObject(robot, sw_obj):
    _run_named_action(robot, sw_obj, "PickupObject", "Picking up object")


def SwitchOn(robot, sw_obj):
    _run_named_action(robot, sw_obj, "SwitchOn", "Switching on object")


def SwitchOff(robot, sw_obj):
    _run_named_action(robot, sw_obj, "SwitchOff", "Switching off object")


def OpenObject(robot, sw_obj):
    _run_named_action(robot, sw_obj, "OpenObject", "Opening object")


def CloseObject(robot, sw_obj):
    _run_named_action(robot, sw_obj, "CloseObject", "Closing object")


def BreakObject(robot, sw_obj):
    _run_named_action(robot, sw_obj, "BreakObject", "Breaking object")


def SliceObject(robot, sw_obj):
    _run_named_action(robot, sw_obj, "SliceObject", "Slicing object")


def ThrowObject(robot, sw_obj):
    _run_named_action(robot, sw_obj, "ThrowObject", "Throwing object")


def CleanObject(robot, sw_obj):
    _run_named_action(robot, sw_obj, "CleanObject", "Cleaning object")


def WashObject(robot, sw_obj):
    _run_named_action(robot, sw_obj, "WashObject", "Washing object")


def DirtyObject(robot, sw_obj):
    _run_named_action(robot, sw_obj, "DirtyObject", "Dirtying object")


def DropHandObject(robot, sw_obj):
    _run_named_action(robot, sw_obj, "DropHandObject", "Dropping hand object")


def PushObject(robot, sw_obj):
    _run_named_action(robot, sw_obj, "PushObject", "Pushing object")


def PullObject(robot, sw_obj):
    _run_named_action(robot, sw_obj, "PullObject", "Pulling object")
   
    
