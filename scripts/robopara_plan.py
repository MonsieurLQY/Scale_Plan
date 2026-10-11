"""Deterministic timed-plan boundary; no arm selection or semantic repair."""
import json
import math
import re

ALIASES = {"flat_newspaper_trash": "trash", "red_tipped_cup": "cup"}
OBJECTS = {"desktop_surface", "marker", "pen_holder", "trash", "trash_bin", "mouse", "laptop", "cup"}
ARMS = {"robot1": "left", "robot2": "right"}
EPSILON = 1e-6
LINE = re.compile(r"([0-9]+(?:\.[0-9]+)?):\s*\(([^()]+)\)\s*\[([0-9]+(?:\.[0-9]+)?)\]")


def canonical(value):
    return ALIASES.get(value, value)


def json_response(text):
    value = text.strip()
    if value.startswith('```') and value.endswith('```'):
        value = '\n'.join(value.splitlines()[1:-1])
    return json.loads(value)


def validate_scene(value):
    if not isinstance(value, dict) or set(value) != {"objects", "relations", "arms"}:
        raise ValueError("Scene requires objects, relations and arms")
    if not isinstance(value['objects'], list) or not isinstance(value['relations'], list) or not isinstance(value['arms'], dict):
        raise ValueError("Invalid scene fields")
    seen = set()
    for obj in value['objects']:
        if not isinstance(obj, dict) or not isinstance(obj.get('id'), str):
            raise ValueError("Invalid visual object")
        obj['id'] = canonical(obj['id'])
        if obj['id'] in seen or obj['id'] not in OBJECTS:
            raise ValueError("Duplicate or unsupported object ID")
        seen.add(obj['id'])
        if obj.get('visible') is not True and obj.get('visible') is not False and obj.get('visible') != 'unknown':
            raise ValueError("visible must be boolean or unknown")
        if not isinstance(obj.get('state'), dict) or not isinstance(obj.get('position'), str):
            raise ValueError("Object requires state and qualitative position")
    return value


def timed_plan(text, scene):
    """Preserve all nonoverlap order; reject malformed/conflicting plans."""
    lines = text.strip().splitlines()
    if lines and lines[0].strip().startswith('```') and lines[-1].strip() == '```':
        lines = lines[1:-1]
    nodes, times, locks = [], [], []
    visible = {o['id'] for o in scene['objects'] if o['visible'] is True}
    for raw in lines:
        if not raw.strip() or raw.lstrip().startswith(';'):
            continue
        match = LINE.fullmatch(raw.strip())
        if not match:
            raise ValueError(f"Invalid timed action: {raw}")
        start, duration = float(match[1]), float(match[3])
        if not math.isfinite(start + duration) or duration <= EPSILON:
            raise ValueError("Nonfinite or nonpositive action duration")
        fields = match[2].lower().split()
        skill, args = fields[0], fields[1:]
        count = {'pick': 3, 'place': 3, 'handover': 3, 'flap_close': 2, 'adjust': 3}
        if skill not in count or len(args) != count[skill]:
            raise ValueError("Unknown skill or wrong action arity")
        if args[0] not in ARMS:
            raise ValueError("Unknown arm")
        arm = ARMS[args[0]]
        if skill == 'handover':
            if args[1] not in ARMS or args[1] == args[0]:
                raise ValueError("Handover requires distinct donor and receiver")
            values = {'object': canonical(args[2]), 'direction': arm + '_to_' + ARMS[args[1]]}
            agents = ['left', 'right']; objects = {values['object']}
            if values['object'] not in {'trash', 'mouse'}:
                raise ValueError("Unsupported handover object")
        else:
            agents = [arm]
            if skill in {'pick', 'place'}:
                values = dict(arm=arm, source=canonical(args[1]), target=canonical(args[2]))
                objects = {values['source'], values['target']}
                if skill == 'pick' and (values['source'] != 'desktop_surface' or values['target'] not in {'marker','trash','mouse','cup'}):
                    raise ValueError("Unsupported pick")
                if skill == 'place' and (values['source'],values['target']) not in {('marker','pen_holder'),('trash','trash_bin'),('mouse','laptop')}:
                    raise ValueError("Unsupported place")
            else:
                values = dict(arm=arm, object=canonical(args[1]))
                objects = {values['object']}
                if skill == 'adjust':
                    values['target_state'] = args[2]
                    if (values['object'], args[2]) != ('cup', 'upright'):
                        raise ValueError("Unsupported adjust")
                elif values['object'] != 'laptop':
                    raise ValueError("Unsupported flap_close")
        if not objects <= visible:
            raise ValueError(f"Action references visually unconfirmed objects: {sorted(objects-visible)}")
        node_id = f'n{len(nodes)+1}'
        nodes.append(dict(node_id=node_id, skill=skill, agents=agents, args=values))
        times.append(dict(node_id=node_id, start=start, duration=duration, end=start+duration))
        locks.append(set(agents) | (objects - {'desktop_surface'}))
    if not nodes:
        raise ValueError('Empty plan')
    edges = []
    for i, left in enumerate(times):
        for j, right in enumerate(times):
            if i == j:
                continue
            if left['end'] <= right['start'] + EPSILON:
                edges.append([nodes[i]['node_id'], nodes[j]['node_id']])
            elif i < j and right['end'] > left['start'] + EPSILON and locks[i] & locks[j]:
                raise ValueError(f'Overlapping exclusive resources: {i+1}, {j+1}')
    # Downstream repeats ownership/precondition checks against the real scene.
    holding = {'left': None, 'right': None}
    for i in sorted(range(len(nodes)), key=lambda n: (times[n]['start'], n)):
        node = nodes[i]; a = node['args']; skill = node['skill']
        if skill == 'handover':
            donor, receiver = a['direction'].split('_to_')
            if holding[donor] != a['object'] or holding[receiver] is not None:
                raise ValueError('Invalid handover ownership')
            holding[receiver], holding[donor] = holding[donor], None
        elif skill == 'pick':
            if holding[a['arm']] is not None or a['target'] in holding.values():
                raise ValueError('Invalid pick ownership')
            holding[a['arm']] = a['target']
        elif skill == 'place':
            if holding[a['arm']] != a['source']:
                raise ValueError('Place without ownership')
            holding[a['arm']] = None
        elif holding[a['arm']] is not None:
            raise ValueError('Skill requires empty arm')
    return dict(task_id='desktop_cleanup', nodes=nodes, precedence=edges), times
