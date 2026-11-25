#!/usr/bin/env python3
"""Write or summarize an EXL3_HOST_EXPERTS plan: which MoE layers keep their routed experts in pinned host RAM.

The plan is {"model.layers.N.ffn.experts": [expert ids kept in host RAM]}. A layer whose list is empty is loaded the
stock vllm_exl3 way: all 384 of its experts live in VRAM, the CPU tier and the VRAM mirror cache skip it, and the DMA
prefetch chain links past it. pack/prepare_pack.py writes the default plan (all 40 layers on the host). This script
writes a separate plan file, so the overlay pack itself stays unmodified and `verify-pack` keeps passing.

  make_host_plan.py --vram-layers 6 --out /models/host-plan-a6000.json       first 6 MoE layers in VRAM
  make_host_plan.py --vram-layers 0,1,2,20,38,39 --out plan.json              explicit layer indices
  make_host_plan.py --summary plan.json                                       JSON summary (used by the entrypoint)

Size estimates use the measured D119 numbers: 15,360 host experts pinned 190.5 GiB, so one expert is ~12.70 MiB and one
full MoE layer (384 experts) is ~4.76 GiB. They are estimates, not measurements on your host.
"""
import argparse
import json
import math
import sys
from pathlib import Path

N_LAYERS = 40
N_EXPERTS = 384
PINNED_ALL_GIB = 190.5                       # measured (README "Where the model lives")
GIB_PER_EXPERT = PINNED_ALL_GIB / (N_LAYERS * N_EXPERTS)
EXTRA_NEED_GIB = 24.5                        # the stock preflight asks for 215 GiB = 190.5 pinned + 24.5


def key(i):
    return f"model.layers.{i}.ffn.experts"


def parse_layers(spec):
    """'6' -> first 6 layers; '0,1,5-7' -> those indices."""
    spec = spec.strip()
    if spec.isdigit() and "," not in spec and "-" not in spec:
        n = int(spec)
        if not 0 <= n <= N_LAYERS:
            raise ValueError(f"layer count must be 0..{N_LAYERS}, got {n}")
        return list(range(n))
    out = set()
    for part in spec.split(","):
        a, _, b = part.strip().partition("-")
        lo, hi = int(a), int(b or a)
        if lo > hi:
            raise ValueError(f"bad range {part!r}")
        out.update(range(lo, hi + 1))
    bad = [i for i in out if not 0 <= i < N_LAYERS]
    if bad:
        raise ValueError(f"layer indices must be 0..{N_LAYERS - 1}, got {sorted(bad)}")
    return sorted(out)


def build(vram_layers):
    vram = set(vram_layers)
    # Empty lists, not missing keys: vllm_exl3's _host_plan_for() falls back to suffix matching on a missing key, while
    # an empty list is an exact hit that means "no host experts for this layer".
    return {key(i): ([] if i in vram else list(range(N_EXPERTS))) for i in range(N_LAYERS)}


def summarize(plan):
    host_layers = [k for k, v in plan.items() if v]
    host_experts = sum(len(set(v)) for v in plan.values())
    partial = [k for k, v in plan.items() if v and len(set(v)) != N_EXPERTS]
    vram_experts = N_LAYERS * N_EXPERTS - host_experts
    pinned = host_experts * GIB_PER_EXPERT
    return {
        "host_layers": len(host_layers),
        "vram_layers": sorted(int(k.split(".")[2]) for k, v in plan.items() if not v),
        "partial_layers": len(partial),
        "host_experts": host_experts,
        "pinned_host_gib": round(pinned, 1),
        "resident_vram_experts_gib": round(vram_experts * GIB_PER_EXPERT, 1),
        "need_gb": math.ceil(pinned + EXTRA_NEED_GIB),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--vram-layers", help="layer count (first N) or comma list / ranges of MoE layer indices for VRAM")
    g.add_argument("--summary", type=Path, help="print a JSON summary of an existing plan")
    p.add_argument("--out", type=Path, help="output plan path (required with --vram-layers)")
    p.add_argument("--field", help="with --summary: print only this field")
    a = p.parse_args(argv)
    if a.summary:
        s = summarize(json.loads(a.summary.read_text()))
        print(s[a.field] if a.field else json.dumps(s))
        return 0
    if not a.out:
        p.error("--out is required with --vram-layers")
    plan = build(parse_layers(a.vram_layers))
    a.out.write_text(json.dumps(plan))
    s = summarize(plan)
    print(json.dumps(s), file=sys.stderr)
    if s["partial_layers"]:
        print("WARNING: partial layers are not supported by the CPU tier / VRAM cache", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
