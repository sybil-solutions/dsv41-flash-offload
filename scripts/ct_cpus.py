#!/usr/bin/env python3
"""Pick CPUs for the AVX2 CPU expert tier: one logical CPU per physical core, skipping the first SKIP cores.

The measured host (24-core EPYC, SMT on) used DSV41_CT_CPUS=2-23 and DSV41_CT_THREADS=22: cores 0-1 left to the engine,
one thread per remaining core, SMT siblings unused. This reproduces that rule on any host and respects the container's
CPU affinity (--cpuset-cpus). Prints "<comma list of cpus> <count>", e.g. "2,3,4,...,15 14".
"""
import os
import sys
from pathlib import Path


def pick(allowed, topo, skip):
    """allowed: set of cpu ids; topo: {cpu: (package, core)}; returns the first allowed cpu of each core minus `skip` cores."""
    seen, firsts = set(), []
    for cpu in sorted(allowed):
        core = topo.get(cpu, (0, cpu))
        if core not in seen:
            seen.add(core)
            firsts.append(cpu)
    return firsts[skip:] if len(firsts) > skip else firsts[-1:]


def topology(cpus):
    topo = {}
    for cpu in cpus:
        d = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            topo[cpu] = (int((d / "physical_package_id").read_text()), int((d / "core_id").read_text()))
        except (OSError, ValueError):
            topo[cpu] = (0, cpu)
    return topo


def main():
    skip = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    allowed = os.sched_getaffinity(0)
    cpus = pick(allowed, topology(allowed), skip)
    print(",".join(map(str, cpus)), len(cpus))


if __name__ == "__main__":
    main()
