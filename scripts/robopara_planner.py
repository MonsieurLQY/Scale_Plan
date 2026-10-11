"""Visual input adapter around the original Scale-Plan multi-stage algorithm."""
import json
from pathlib import Path
import time

from Scale_Plan_DAG import (ScalePlanTaskManager, build_stage1_prompt, build_dag_edges,
                            dfs_action_chain, extract_required_parameters)
from robopara_plan import json_response, validate_scene, canonical

ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


class GeminiHandler:
    allow_format_repair = False

    def __init__(self, client, directory):
        self.client = client
        self.directory = directory
        self.context = ''
        self.stage = 'initialization'
        self.index = 0

    def query_model(self, prompt, gpt_version=None, **kwargs):
        if isinstance(prompt, list):
            prompt = '\n\n'.join(f"{m['role']}: {m['content']}" for m in prompt)
        full = self.context + '\n\n' + prompt
        self.index += 1
        prefix = self.directory / f'{self.index:02d}_{self.stage}'
        prefix.with_suffix('.prompt.txt').write_text(full)
        start = time.perf_counter()
        try:
            text = self.client.call(full)
            prefix.with_suffix('.response.txt').write_text(text)
            return {}, text
        finally:
            save(prefix.with_suffix('.metadata.json'), {
                'stage': self.stage, 'elapsed_s': time.perf_counter()-start,
                'model_calls': getattr(self.client, 'calls', [])})


class RoboParaPlanner(ScalePlanTaskManager):
    def __init__(self, client, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        handler = GeminiHandler(client, self.directory)
        super().__init__(str(ROOT), client.model, '',
                         prompt_decompse_set='robopara/decompose',
                         prompt_allocation_set='robopara/allocation', verbose=False,
                         llm_handler=handler, resources_path=ROOT/'resources/robopara',
                         logs_root=self.directory/'internal', clean_generated=False, use_pddl_solver=False)

    def perceive(self, image):
        prompt = '''Inspect this actual main-camera image of a dual-arm desktop scene.
Return ONLY JSON with objects (list), relations (list of strings), arms (object).
Each object: {"id": canonical ID, "visible": true/false/"unknown",
"position": qualitative image/physical position, "state": {property: observed value or "unknown"}}.
Canonical vocabulary: desktop_surface, marker (pen), pen_holder, trash (flat newspaper),
trash_bin, mouse, laptop, cup (red tipped cup). Do not invent detections.
Report visible objects, their positions and states (open/closed, upright/tipped, on/in).
Describe both robot1/left and robot2/right grippers and apparent held objects, or unknown.
Physical left/right is from bases facing workspace, preserved in this camera.
Do not produce a task plan. Ambiguous and occluded states must be unknown, not false.
'''
        (self.directory/'perception.prompt.txt').write_text(prompt)
        raw = self.llm.client.call(prompt, image_path=image)
        (self.directory/'perception.response.txt').write_text(raw)
        scene = validate_scene(json_response(raw))
        save(self.directory/'scene.json', scene)
        return scene

    def create_DAG_and_filter_environment(self, task, objects_ai, objects_ai_names, plot_DAG=False):
        # Preserve the original SR/RR action graph and backward traversal.
        edges = build_dag_edges(self.domain.actions, mode='both')
        self.llm.stage = 'terminal_actions'
        prompt = build_stage1_prompt(task, self.domain_content, self.domain.actions, objects_ai_names)
        _, raw = self.llm.query_model(prompt + '\nReturn {"actions": [{"action": "place", "parameters": {"target": "canonical_id"}}]}. Use actual necessary terminal actions, not this example.')
        value = json_response(raw)
        actions = value.get('actions') if isinstance(value, dict) else value
        if not isinstance(actions, list) or not actions:
            raise ValueError('No terminal actions')
        for a in actions:
            if not isinstance(a, dict) or a.get('action') not in self.domain.actions or not isinstance(a.get('parameters'), dict):
                raise ValueError('Invalid terminal action')
        chain = dfs_action_chain([a['action'] for a in actions], edges)
        types = extract_required_parameters(chain, self.domain.actions)
        mentioned = {canonical(str(v)) for a in actions for v in a['parameters'].values()}
        # Object is the common PDDL type in this domain: retain all visible instances
        # of required types, never collapse multiple instances by category.
        self.filtered_env = [o for o in objects_ai if o['objectType'] in types or o['objectId'] in mentioned]
        names = {o['objectId']: o['objectId'] for o in self.filtered_env}
        self.filtered_objects_category_str = 'objects = ' + json.dumps(list(names)) + '\nVisual scene = ' + json.dumps(self.scene)
        self.filtered_env_names_str = 'objects = ' + json.dumps(names)
        save(self.directory/'action_graph.json', dict(edges=sorted(edges), terminal_actions=actions, action_chain=chain, filtered_objects=list(names)))

    def plan(self, instruction, scene):
        self.scene = scene
        self.llm.context = (ROOT/'scripts/llm_prompts/robopara/decompose.txt').read_text() + '\nVisual evidence:\n' + json.dumps(scene) + '\nUser task intent:\n' + instruction
        objects = [dict(objectId=o['id'], objectType='object') for o in scene['objects'] if o['visible'] is True]
        robots = [dict(name=f'robot{i}', skills=list(self.domain.actions), mass_capacity=100) for i in (1,2)]
        self.create_DAG_and_filter_environment(instruction, objects, {o['objectId']:o['objectId'] for o in objects})
        self.llm.stage = 'decomposition'
        decomposed = self._generate_decomposed_plan(instruction, self.domain_content, robots, self.filtered_objects_category_str)
        self.llm.stage = 'allocation'
        allocated = self._generate_allocation_plan(decomposed, robots, self.filtered_objects_category_str)
        self.llm.stage = 'summary'
        summary = self._generate_problem_summary(decomposed, allocated, robots)
        self.llm.stage = 'subplans'
        subplans = self._generate_problem_files(robots, summary, self.filtered_objects_category_str)
        if not subplans:
            raise ValueError('No subplans generated')
        self.llm.stage = 'merge'
        # Keep the original merge operation, making its generated subplans explicit.
        combined = self._combine_all_plans(0, allocated + '\nGenerated subplans:\n' + '\n'.join(subplans), '')
        self.llm.stage = 'reference_binding'
        matched = self._match_references_for_plan(combined, self.filtered_env_names_str)
        (self.directory/'timed_plan.txt').write_text(matched)
        return matched
