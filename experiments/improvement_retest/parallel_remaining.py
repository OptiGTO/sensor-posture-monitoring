"""Resume independent RF splits without changing frozen training algorithms."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import json
import subprocess
import sys

from .resume_frozen import prepare_execution, digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    out = parser.parse_args().output.resolve()
    source = prepare_execution(out)
    protocol = json.loads((out / "protocol.json").read_text())
    partitions = protocol["dataset_partitions"]
    if isinstance(partitions, dict):
        partitions = list(partitions.values())
    # Only the scheduler's partition iterator is narrowed. The exact frozen
    # worker still verifies protocol/code/data and runs its original fit loop.
    worker = """import sys
from pathlib import Path
from experiments.improvement_retest import run
original_partitions = run.partitions
partition_id = sys.argv[2]
run.partitions = lambda p: [s for s in original_partitions(p) if s['id'] == partition_id]
assert len(run.partitions(run.verify(Path(sys.argv[1])))) == 1
run.development_worker(Path(sys.argv[1]), 'random_forest')
"""
    record = json.loads((out / "partition_resume.json").read_text())
    record.update({"launcher_sha256": digest(Path(__file__)), "worker_code": worker,
                   "partition_ids": [p["id"] for p in partitions],
                   "source": str(source), "concurrent_workers": 3})
    (out / "partition_resume.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")

    def one(part):
        log = out / "logs" / f"parallel_random_forest_{part['id']}.log"
        with log.open("a") as stream:
            subprocess.run([sys.executable, "-c", worker, str(out), part["id"]],
                           cwd=source, stdout=stream, stderr=subprocess.STDOUT, check=True)
        print("RF PARTITION COMPLETE", part["id"], flush=True)

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(one, partitions))


if __name__ == "__main__":
    main()
