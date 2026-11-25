import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).parents[1]


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


plan = load('make_host_plan', 'pack/make_host_plan.py')
ct = load('ct_cpus', 'scripts/ct_cpus.py')


class HostPlanTest(unittest.TestCase):
    def test_zero_vram_layers_matches_prepare_pack_default(self):
        # prepare_pack.py writes {model.layers.i.ffn.experts: list(range(384)) for i in range(40)}
        default = {f"model.layers.{i}.ffn.experts": list(range(384)) for i in range(40)}
        self.assertEqual(plan.build([]), default)
        self.assertEqual(json.dumps(plan.build([])), json.dumps(default))   # byte-identical -> same sha256 as the pack
        s = plan.summarize(default)
        self.assertEqual((s['host_layers'], s['need_gb'], s['pinned_host_gib']), (40, 215, 190.5))

    def test_six_vram_layers(self):
        p = plan.build(plan.parse_layers('6'))
        s = plan.summarize(p)
        self.assertEqual(s['host_layers'], 34)
        self.assertEqual(s['vram_layers'], [0, 1, 2, 3, 4, 5])
        self.assertEqual(s['partial_layers'], 0)
        self.assertAlmostEqual(s['pinned_host_gib'], 161.9, places=1)
        self.assertAlmostEqual(s['resident_vram_experts_gib'], 28.6, places=1)
        self.assertEqual(p['model.layers.0.ffn.experts'], [])      # explicit empty list, not a missing key
        self.assertEqual(len(p), 40)

    def test_explicit_layers_and_ranges(self):
        self.assertEqual(plan.parse_layers('0,5-7,39'), [0, 5, 6, 7, 39])
        for bad in ('41', '40-41', '7-5', '-1'):
            with self.assertRaises(ValueError):
                plan.parse_layers(bad)

    def test_partial_layer_is_flagged(self):
        p = plan.build([])
        p['model.layers.3.ffn.experts'] = list(range(100))
        self.assertEqual(plan.summarize(p)['partial_layers'], 1)

    def test_cli_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'p.json'
            self.assertEqual(plan.main(['--vram-layers', '5', '--out', str(out)]), 0)
            self.assertEqual(plan.summarize(json.loads(out.read_text()))['host_layers'], 35)


class CtCpusTest(unittest.TestCase):
    def test_measured_host_rule(self):
        # 24-core EPYC with SMT: cpu n and n+24 share core n -> 2..23, 22 threads (the measured DSV41_CT_CPUS=2-23)
        topo = {c: (0, c % 24) for c in range(48)}
        self.assertEqual(ct.pick(set(range(48)), topo, 2), list(range(2, 24)))

    def test_threadripper_7970x(self):
        # 32 cores / 64 threads: cpu n and n+32 share core n -> 2..31, 30 threads
        topo = {c: (0, c % 32) for c in range(64)}
        self.assertEqual(ct.pick(set(range(64)), topo, 2), list(range(2, 32)))

    def test_16_core_desktop_and_cpuset(self):
        topo = {c: (0, c % 16) for c in range(32)}
        self.assertEqual(ct.pick(set(range(32)), topo, 2), list(range(2, 16)))
        self.assertEqual(ct.pick({0, 1, 2, 3, 16, 17}, topo, 2), [2, 3])

    def test_tiny_host_keeps_one_cpu(self):
        self.assertEqual(ct.pick({0, 1}, {0: (0, 0), 1: (0, 1)}, 2), [1])


if __name__ == '__main__':
    unittest.main()
