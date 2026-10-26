"""One side of an A/B: load the model from whichever mini_vllm is first on PYTHONPATH, run the
workloads, and print one JSON line of results. cloud.app's ab runs this once per version per
round, in one container, so both versions see the same host.

    python -m cloud.ab_worker <backend> '<json list of Workload fields>'
"""

from __future__ import annotations

import json
import statistics
import sys

from cloud import adapters, metrics
from cloud import workload as wl


def main() -> None:
    backend, specs = sys.argv[1], json.loads(sys.argv[2])
    factory = adapters.MiniVllmFactory(backend)
    results = {}
    for spec in specs:
        workload = wl.Workload(**spec)
        adapter = factory.fresh()
        records, wall = wl.run(adapter, workload)
        summary = metrics.summarize(records, wall)
        results[workload.name] = {
            "output_tok_s": summary["output_tok_s"],
            "tpot_p50_ms": summary["tpot_ms"]["p50"] if summary["tpot_ms"] else None,
            "step_p50_ms": statistics.median(seconds for seconds, _ in adapter.steps) * 1e3,
        }
        factory.release(adapter)
    print(json.dumps({"load_s": factory.load_s, "python_loop_ms": metrics.python_loop_ms(),
                      "workloads": results}))


if __name__ == "__main__":
    main()
