import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from robopara_plan import timed_plan, validate_scene
from robopara_planner import RoboParaPlanner


def scene():
    return {'objects':[dict(id=o, visible=True, position='unknown', state={'status':'unknown'})
                       for o in ('desktop_surface','marker','pen_holder','trash','trash_bin','laptop','mouse','cup')],
            'relations':[], 'arms':{'robot1':'unknown','robot2':'unknown'}}


class PlanTests(unittest.TestCase):
    def test_parallel_preserved(self):
        p,t = timed_plan('0: (pick robot1 desktop_surface marker) [4]\n0: (flap_close robot2 laptop) [8.5]\n4: (place robot1 marker pen_holder) [4.5]', scene())
        self.assertEqual(p['precedence'], [['n1','n3']])
        self.assertEqual(t[1]['start'], 0)

    def test_handover(self):
        p,_ = timed_plan('0: (pick robot1 desktop_surface flat_newspaper_trash) [4]\n4: (handover robot1 robot2 trash) [8]\n12: (place robot2 trash trash_bin) [4.5]', scene())
        self.assertEqual(p['nodes'][1]['args']['direction'],'left_to_right')
        self.assertEqual(p['nodes'][1]['agents'], ['left','right'])

    def test_reject_without_repair(self):
        cases = ['0: (pick robot1 desktop_surface marker) [0]',
                 '-1: (pick robot1 desktop_surface marker) [4]',
                 'nan: (pick robot1 desktop_surface marker) [4]',
                 '0: (handover robot1 robot2 trash) [8]',
                 '0: (pick robot1 marker) [4]',
                 '0: (place robot1 marker pen_holder) [4]',
                 '0: (pick robot1 desktop_surface marker) [4]\n1: (flap_close robot1 laptop) [8.5]',
                 '0: (pick robot1 desktop_surface trash) [4]\n1: (pick robot2 desktop_surface trash) [4]']
        for raw in cases:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                timed_plan(raw, scene())

    def test_unknown_visibility_not_instruction_evidence(self):
        value = scene(); value['objects'][1]['visible'] = 'unknown'
        with self.assertRaisesRegex(ValueError, 'visually unconfirmed'):
            timed_plan('0: (pick robot1 desktop_surface marker) [4]', value)
        self.assertEqual(validate_scene(value)['objects'][1]['visible'], 'unknown')

    def test_duplicate_alias_and_instances(self):
        value=scene(); value['objects'].append(dict(id='flat_newspaper_trash',visible=True,position='left',state={}))
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            validate_scene(value)
        self.assertEqual(len(validate_scene(scene())['objects']),8)


class FakeGemini:
    model = 'gemini-test'
    calls = []
    def __init__(self):
        self.prompts=[]
        self.responses=iter([
            json.dumps({'actions':[{'action':'place','parameters':{'target':'pen_holder'}}]}),
            'Subtask: store marker',
            '**Assigned Robot**: robot1\n**Objects Involved**: marker, pen_holder, desktop_surface',
            '# Problem content summary:\n#SubTask 1: store marker\n**Assigned Robot**: robot1\n**Objects Involved**: marker, pen_holder, desktop_surface\n# Sequence of Operations:\npick then place',
            '0: (pick robot1 desktop_surface marker) [4]\n4: (place robot1 marker pen_holder) [4.5]',
            '0: (pick robot1 desktop_surface marker) [4]\n4: (place robot1 marker pen_holder) [4.5]',
            '0: (pick robot1 desktop_surface marker) [4]\n4: (place robot1 marker pen_holder) [4.5]'])
    def call(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return next(self.responses)


class PipelineTests(unittest.TestCase):
    def test_original_stages_with_injected_model(self):
        for instruction in ['Store the marker.', '请把笔放进笔筒。']:
            with tempfile.TemporaryDirectory() as d:
                client=FakeGemini(); planner=RoboParaPlanner(client,d)
                raw=planner.plan(instruction,scene())
                p,_=timed_plan(raw,scene())
                self.assertEqual(len(p['nodes']),2)
                self.assertEqual(len(client.prompts),7)
                stages=[json.loads(p.read_text())['stage'] for p in sorted(Path(d).glob('*.metadata.json'))]
                self.assertEqual(stages,['terminal_actions','decomposition','allocation','summary','subplans','merge','reference_binding'])
                self.assertTrue(all(instruction in prompt for prompt in client.prompts))
                graph=json.loads((Path(d)/'action_graph.json').read_text())
                self.assertIn('pick',graph['action_chain'])
                self.assertIn('desktop_surface',graph['filtered_objects'])
                self.assertIn('marker',graph['filtered_objects'])
                self.assertIn('pen_holder',graph['filtered_objects'])
                self.assertTrue(all('selected_goals' not in prompt for prompt in client.prompts))

if __name__=='__main__':
    unittest.main()
