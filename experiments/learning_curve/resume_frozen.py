"""Resume an interrupted study from its immutable source and data snapshots."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import shutil
import subprocess
import sys

from experiments.learning_curve.run import ROOT, read, digest, now, protected_snapshot
from src.model_comparison import write_json


def verify_evidence(out):
    p = read(out / "protocol.json")
    if digest(out / "protocol.json") != read(out / "protocol_lock.json")["sha256"]:
        raise ValueError("Protocol changed")
    if digest(out / "data.npz") != p["cache_sha256"]:
        raise ValueError("Cache changed")
    for relative, sha in p["source_hashes"].items():
        if digest(out / "source_snapshot" / relative) != sha:
            raise ValueError(f"Frozen evidence changed: {relative}")
    return p


def verify_originals(out):
    before, current = read(out / "protected_before.json"), protected_snapshot()
    if any(current.get(k) != v for k, v in before.items()):
        raise ValueError("An original protected file changed or disappeared")
    added = {k: v for k, v in current.items() if k not in before}
    if added:
        recorded = read(out / "source_drift.json")["external_added_files"]
        if added != recorded:
            raise ValueError("External added files changed since recovery audit")
    return added


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True)
    out = parser.parse_args().output.resolve(); p = verify_evidence(out)
    work = out / "execution_snapshot"
    if not work.exists():
        work.mkdir()
        # Copy import dependencies, then replace every frozen module with its original bytes.
        shutil.copytree(ROOT / "src", work / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        shutil.copy2(ROOT / "validate_personalization.py", work / "validate_personalization.py")
        package = work / "experiments/learning_curve"; package.mkdir(parents=True)
        for relative in p["source_hashes"]:
            dest = work / relative; dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(out / "source_snapshot" / relative, dest)
        previous = Path(p["full_cohort_source"])
        shutil.copy2(previous / "source_snapshot/src/app_services.py", work / "src/app_services.py")
        before, current = read(out / "protected_before.json"), protected_snapshot()
        write_json(out / "source_drift.json", {
            "detected_at": now(), "reason": "Concurrent workspace changes interrupted the strict live-source hash check",
            "changed_live_sources": [r for r, sha in p["source_hashes"].items() if digest(ROOT/r) != sha],
            "external_added_files": {k:v for k,v in current.items() if k not in before},
            "recovery": "Continue with frozen source bytes in execution_snapshot; reuse audited cache and completed fits",
            "execution_source_hashes": {str(f.relative_to(work)):digest(f) for f in work.rglob("*.py")}})
    audit = read(out / "source_drift.json")
    for relative, sha in audit["execution_source_hashes"].items():
        if digest(work / relative) != sha: raise ValueError("Execution snapshot changed")
    verify_originals(out)
    jobs = [(f,s["id"]) for f in p["models"] for s in p["splits"]
            if any(not (out/"fits"/f/s["id"]/a["id"]/"metrics.json").exists() for a in s["arms"])]
    def launch(job):
        f,s = job
        with (out/"logs"/f"{f}_{s}.log").open("a") as log:
            subprocess.run([sys.executable,"-m","experiments.learning_curve.integrated","--output",str(out),"--worker",f,s],
                           cwd=work,stdout=log,stderr=subprocess.STDOUT,check=True)
        return job
    with ThreadPoolExecutor(max_workers=3) as pool:
        for result in as_completed([pool.submit(launch,j) for j in jobs]):
            print("DONE",*result.result(),flush=True)
    verify_originals(out)
    count=len(list((out/"fits").rglob("metrics.json")))
    if count != 366: raise ValueError(f"Incomplete fits: {count}")
    write_json(out/"training_complete.json",{"finished_at":now(),"fits":count,
        "original_protected_files_unchanged":True,"external_added_files":audit["external_added_files"],
        "frozen_source_recovery":True})


if __name__ == "__main__": main()
