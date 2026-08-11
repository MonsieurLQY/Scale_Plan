"""Finalize execution and evaluate task results.

This file is appended to the generated executable by ``execute_plan.py``.
Runtime state, imports, and helpers are intentionally supplied by the earlier
``imports_aux_fn.py``, ``aithor_connect.py``, and generated-plan fragments.
"""


def _state_satisfied(objects, object_name, state):
    """Return whether any matching object has the requested AI2-THOR state."""
    state = str(state).upper()
    matching_objects = [
        obj for obj in objects if object_name.lower() in obj.get("name", "").lower()
    ]

    checks = {
        "SLICED": lambda obj: obj.get("isSliced", False),
        "OFF": lambda obj: not obj.get("isToggled", False),
        "ON": lambda obj: obj.get("isToggled", False),
        "HOT": lambda obj: obj.get("temperature") == "Hot",
        "COOKED": lambda obj: obj.get("isCooked", False),
        "OPENED": lambda obj: obj.get("isOpen", False),
        "CLOSED": lambda obj: not obj.get("isOpen", False),
        "PICKED": lambda obj: obj.get("isPickedUp", False),
        "CLEANED": lambda obj: not obj.get("isDirty", False),
        "DIRTY": lambda obj: obj.get("isDirty", False),
        "BROKEN": lambda obj: obj.get("isBroken", False),
    }
    check = checks.get(state)
    return bool(check and any(check(obj) for obj in matching_objects))


def _count_satisfied_contents(objects, object_name, expected_contents, required_count):
    """Count requested objects contained by a matching receptacle."""
    receptacle_ids = {
        contained_id.lower()
        for obj in objects
        if object_name.lower() in obj.get("name", "").lower()
        for contained_id in (obj.get("receptacleObjectIds") or [])
    }

    satisfied = 0
    for expected_object in expected_contents:
        if any(expected_object.lower() in object_id for object_id in receptacle_ids):
            satisfied += 1
            if satisfied >= required_count:
                break
    return satisfied


def _evaluate_ground_truth(objects, conditions):
    total_conditions = 0
    completed_conditions = 0
    messages = []

    for condition in conditions:
        object_name = condition["name"]
        state = condition.get("state")
        expected_contents = condition.get("contains") or []
        required_contents = condition.get("num_contains", len(expected_contents))

        print("Ground truth condition:", condition)
        if state and str(state).upper() != "NONE":
            total_conditions += 1
            if _state_satisfied(objects, object_name, state):
                completed_conditions += 1
                message = f"{str(state).upper()} condition met for object: {object_name}"
                print(message)
                messages.append(message)

        total_conditions += required_contents
        satisfied_contents = _count_satisfied_contents(
            objects,
            object_name,
            expected_contents,
            required_contents,
        )
        completed_conditions += satisfied_contents
        if satisfied_contents:
            message = (
                f"CONTAINS condition met for object: {object_name} "
                f"({satisfied_contents}/{required_contents})"
            )
            print(message)
            messages.append(message)

    return total_conditions, completed_conditions, messages


# Allow the action-processing thread to drain before ending the simulation.
for _ in range(25):
    action_queue.extend(({"action": "Done"},) * 3)
    time.sleep(0.1)

task_over = True
time.sleep(5)

executability = success_exec / total_exec if total_exec else 1.0
print("Success Exec:", success_exec, "Total Exec:", total_exec)
print("Executability:", executability)

if verbose:
    print("Ground Truth:", ground_truth, "\n")

objects = list(c.last_event.metadata["objects"])
gcr_tasks, gcr_complete, evaluation_messages = _evaluate_ground_truth(
    objects,
    ground_truth,
)
for message in evaluation_messages:
    log += f"{message}\n"

gcr = gcr_complete / gcr_tasks if gcr_tasks else 1.0
task_completion = int(gcr == 1.0)

print("GCR Tasks:", gcr_tasks, "GCR Completed:", gcr_complete)
print("Task completed successfully." if task_completion else "Task not completed successfully.")

if verbose:
    print(
        f"Task Completion:{task_completion}, Goal Condition Recall:{gcr}, "
        f"Executability:{executability}"
    )

results_content = (
    f"Log:{log}\n"
    f"Successful Executions:{success_exec}, Total Executions:{total_exec}\n"
    f"GCR Completed:{gcr_complete}, GCR Tasks:{gcr_tasks}\n"
    f"Task Completion:{task_completion}, Goal Condition Recall:{gcr}, "
    f"Executability:{executability}, Planningtime:{Planningtime}\n"
)
with open(results_path, "w", encoding="utf-8") as results_file:
    results_file.write(results_content)

generate_video(frame_rate=3)
