#!/usr/bin/env python3
"""Run an image-grounded Scale-Plan episode against RoboPARA's shared executor."""
import argparse
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import uuid

from gemini_client import GeminiClient
from robopara_planner import RoboParaPlanner, save, ROOT
from robopara_plan import timed_plan

PROTOCOL = 'robopara.scaleplan.v1'


class Bridge:
    def __init__(self, url, timeout=1800):
        self.url = url.rstrip('/')
        self.timeout = timeout

    def post(self, operation, **fields):
        data = json.dumps(dict(protocol_version=PROTOCOL, **fields)).encode()
        request = Request(self.url+'/v1/scaleplan/'+operation, data=data, headers={'Content-Type':'application/json'})
        try:
            with urlopen(request, timeout=self.timeout) as response:
                value = json.load(response)
        except HTTPError as exc:
            raise RuntimeError(f'Bridge {operation}: HTTP {exc.code}: {exc.read().decode()}') from exc
        if value.get('protocol_version') != PROTOCOL or 'error' in value:
            raise RuntimeError(f'Bridge {operation} failed: {value}')
        return value

    def frame(self, episode, path):
        with urlopen(self.url+'/v1/scaleplan/frame?episode_id='+episode, timeout=self.timeout) as response:
            path.write_bytes(response.read())

    def result(self, episode):
        with urlopen(self.url+'/v1/scaleplan/result?episode_id='+episode, timeout=self.timeout) as response:
            value = json.load(response)
        if value.get('protocol_version') != PROTOCOL:
            raise ValueError('Protocol mismatch')
        return value['result']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server-url', default='http://127.0.0.1:8040')
    parser.add_argument('--instruction', type=Path, default=ROOT/'instruction.txt')
    parser.add_argument('--goals-config', type=Path, default=ROOT/'configs/desktop_cleanup.json')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'logs/robopara')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--timeout', type=float, default=1800)
    for name in ('credentials','project','location','model','fallback-model'):
        parser.add_argument('--'+name)
    args = parser.parse_args()
    instruction = args.instruction.read_text(encoding='utf-8').strip()
    if not instruction:
        parser.error('instruction is empty')
    config = json.loads(args.goals_config.read_text())
    run_id = uuid.uuid4().hex
    directory = args.output_dir/run_id
    directory.mkdir(parents=True)
    (directory/'instruction.txt').write_text(instruction)
    save(directory/'goals.json', config)
    save(directory/'run.json', dict(run_id=run_id, seed=args.seed, server_url=args.server_url))
    bridge = Bridge(args.server_url, args.timeout)
    episode = None
    phase = 'reset'
    perception_s = planning_s = 0.0
    start = None
    try:
        reset = bridge.post('reset', run_id=run_id, task_id=config['task_id'], seed=args.seed, goals=config['goals'])
        save(directory/'reset.json', reset)
        episode = reset['episode_id']
        bridge.frame(episode, directory/'main.jpg')
        phase = 'perception'
        with GeminiClient(api_path=args.credentials, project=args.project, location=args.location,
                          model=args.model, fallback_model=args.fallback_model) as client:
            planner = RoboParaPlanner(client, directory)
            try:
                start = time.perf_counter()
                scene = planner.perceive(directory/'main.jpg')
                perception_s = time.perf_counter()-start
                phase = 'planning'; start = time.perf_counter()
                raw = planner.plan(instruction, scene)
                planning_s = time.perf_counter()-start
                phase = 'parsing'
                plan, timing = timed_plan(raw, scene)
                save(directory/'plan.json', plan)
                save(directory/'original_timing.json', timing)
            finally:
                save(directory/'model_calls.json', client.calls)
        phase = 'execution'
        result = bridge.post('execute', episode_id=episode, plan=plan,
                             perception_wallclock_s=perception_s, planning_wallclock_s=planning_s)['result']
        save(directory/'result.json', result)
        print(json.dumps(dict(output_dir=str(directory), episode_id=episode,
                              task_success=result['task_success'], steps=result.get('steps'),
                              termination_reason=result['termination_reason'],
                              goal_ratio=result.get('score', {}).get('goal_ratio')), ensure_ascii=False))
        # Skill/planning failures are results; zero only means actual execution occurred.
        return 0 if result.get('steps', 0) > 0 else 2
    except (Exception, KeyboardInterrupt) as exc:
        if start is not None:
            if phase == 'perception': perception_s = time.perf_counter()-start
            if phase == 'planning': planning_s = time.perf_counter()-start
        save(directory/'error.json', dict(phase=phase, error_type=type(exc).__name__, error=str(exc)))
        if episode:
            try:
                if phase == 'execution':
                    result = bridge.result(episode)  # Never retry execute after a transport error.
                else:
                    result = bridge.post('finalize', episode_id=episode, reason=phase+'_failed',
                                         perception_wallclock_s=perception_s, planning_wallclock_s=planning_s)['result']
                save(directory/'result.json', result)
            except Exception as final_error:
                save(directory/'result_query_error.json', dict(error=str(final_error)))
        print(f'{phase} failed: {type(exc).__name__}: {exc}; artifacts: {directory}')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
