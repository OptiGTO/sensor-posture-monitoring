"""Freeze and audit exact TRAIN window/feature label collisions, without fits.

Only chosen numeric NPZ rows are decoded. Whole identity metadata and file bytes
are read for scope and integrity; skipped compressed bytes may be decompressed
internally by ZIP seeking without being interpreted as numeric observations.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
from importlib.metadata import version as package_version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import zipfile
from zoneinfo import ZoneInfo

import numpy as np

VERSION = "exact_input_collision_audit_v1"
ORIGIN = "research/results/2026-09-22_055131_iterative_019"
SEQUENCE_ORIGIN = "research/results/2026-09-22_032600_iterative_014"
REPORT = "docs/reports/2026-09-22_exact_input_collision_diagnostic.md"
MODES = ("exact_bytes", "signed_zero_normalized")
CORE = ("src/posture_core.py", "src/window_pipeline.py", "src/__init__.py",
        "experiments/iterative_improvement/record_training.py", "experiments/iterative_improvement/supplemental.py")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def reference(path, root):
    stat = path.stat()
    return {"path": str(path.relative_to(root)), "sha256": digest(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def array_hash(array):
    value = np.ascontiguousarray(array)
    result = hashlib.sha256(json.dumps([value.dtype.str, list(value.shape)]).encode())
    result.update(value.tobytes())
    return result.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    path.chmod(0o444)


def selected_rows(path, key, indices, expected_rows):
    """Decode sorted selected rows only; support scalar, vector and window rows."""
    take = np.asarray(indices)
    require(take.ndim == 1 and take.dtype.kind in "iu" and len(take) > 0
        and np.all(take[1:] > take[:-1]), "Selected indices must be sorted unique integers")
    with zipfile.ZipFile(path) as archive, archive.open(key + ".npy") as stream:
        version = np.lib.format.read_magic(stream)
        require(version in ((1, 0), (2, 0)), "Unsupported NPY header version")
        reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = reader(stream)
        require(not fortran and not dtype.hasobject and dtype.kind in "fiu" and len(shape) in (1, 2, 3)
            and shape[0] == expected_rows, "Unsupported selected numeric schema")
        require(int(take[0]) >= 0 and int(take[-1]) < shape[0], "Selected index outside archive")
        start = stream.tell()
        row_bytes = int(np.prod(shape[1:])) * dtype.itemsize
        result = np.empty((len(take), *shape[1:]), dtype=dtype)
        boundaries = np.r_[0, np.flatnonzero(take[1:] != take[:-1] + 1) + 1, len(take)]
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            stream.seek(start + int(take[left]) * row_bytes)
            count = int(right - left)
            block = stream.read(count * row_bytes)
            require(len(block) == count * row_bytes, "Truncated selected row block")
            result[left:right] = np.frombuffer(block, dtype=dtype).reshape((count, *shape[1:]))
    return result


def _content_hash(payload):
    return hashlib.sha256(payload).hexdigest()


def collision_summary(values, labels, people, records, origins, mode="exact_bytes"):
    """Group exact payload bytes, confirming equality inside every SHA bucket."""
    values, labels, people, records = map(np.asarray, (values, labels, people, records))
    require(mode in MODES, "Unknown exact equality mode")
    require(values.ndim in (2, 3) and len(values) > 0 and all(s > 0 for s in values.shape)
        and values.dtype == np.dtype(np.float32) and np.isfinite(values).all(), "Expected finite nonempty float32 rows")
    n = len(values)
    require(labels.ndim == people.ndim == records.ndim == 1 and len(labels) == len(people) == len(records) == len(origins) == n
        and labels.dtype.kind in "iu" and np.all(labels >= 0) and np.all(labels < 7), "Invalid aligned labels or identities")
    require(all(isinstance(v, str) and v.strip() for v in [*people.tolist(), *records.tolist()]), "Nonempty string identities required")
    require(all(isinstance(o, dict) and set(o) == {"cache", "index"} and o["cache"] in {"base", "ordinary"}
        and type(o["index"]) is int and o["index"] >= 0 for o in origins), "Invalid origin rows")
    require(len({(o["cache"], o["index"]) for o in origins}) == n, "Repeated origin identity")
    data = np.array(values, dtype=np.float32, order="C", copy=True)
    if mode == "signed_zero_normalized":
        data[data == 0] = np.float32(0.)
    header = json.dumps([data.dtype.str, list(data.shape[1:])]).encode() + b"\0"
    buckets = defaultdict(list)
    for index, row in enumerate(data):
        payload = header + row.tobytes(order="C")
        key = _content_hash(payload)
        for group in buckets[key]:
            if group["payload"] == payload:
                group["indices"].append(index)
                break
        else:
            buckets[key].append({"payload": payload, "indices": [index]})
    groups = [(key, group) for key, bucket in buckets.items() for group in bucket]
    details = []
    for key, group in groups:
        indices = group["indices"]
        if len(indices) < 2:
            continue
        counts = np.bincount(labels[indices].astype(np.int64), minlength=7)
        details.append({"input_sha256": key, "rows": len(indices), "class_counts": counts.tolist(),
            "conflicting_labels": int(np.count_nonzero(counts)) > 1,
            "subjects": len(set(people[indices].tolist())), "records": len(set(records[indices].tolist())),
            "members": [{"row": i, "origin": origins[i], "class_id": int(labels[i]),
                "subject_sha256": _content_hash(str(people[i]).encode()),
                "record_sha256": _content_hash(str(records[i]).encode())} for i in indices]})
    conflicts = [row for row in details if row["conflicting_labels"]]
    same = [row for row in details if not row["conflicting_labels"]]
    return {"mode": mode, "rows": n, "row_shape": list(data.shape[1:]), "dtype": data.dtype.str,
        "unique_inputs": len(groups), "duplicate_groups": len(details), "duplicate_rows": sum(row["rows"] for row in details),
        "excess_duplicate_rows": n - len(groups), "same_label_duplicate_groups": len(same),
        "same_label_duplicate_rows": sum(row["rows"] for row in same), "conflict_groups": len(conflicts),
        "conflicting_rows": sum(row["rows"] for row in conflicts),
        "cross_subject_conflict_groups": sum(row["subjects"] > 1 for row in conflicts),
        "cross_record_conflict_groups": sum(row["records"] > 1 for row in conflicts),
        "sha_bucket_byte_checks": True, "hash_collision_buckets": sum(len(bucket) > 1 for bucket in buckets.values()),
        "group_details": details}


def environment():
    return {"python": platform.python_version(), "numpy": np.__version__, "executable": sys.executable,
        **{name: package_version(name) for name in ("scipy", "scikit-learn", "pandas")}}


def _verify_protected(root, origin):
    protected = read(origin / "protected_before.json")
    require(len(protected) == 1499, "Unexpected protected inventory size")
    for relative, expected in protected.items():
        actual = reference(root / relative, root)
        require({key: actual[key] for key in ("sha256", "size", "mtime_ns")} == expected, "Protected original changed: " + relative)
    return protected


def freeze(root, output, report=None):
    require(not output.exists() and output.is_relative_to(root / "research/results"), "Use a fresh independent research/results directory")
    report = (root / REPORT) if report is None else report.resolve()
    require(report.is_relative_to(root / "docs/reports") and not report.exists(), "Use a new report under docs/reports")
    origin, sequence_origin = root / ORIGIN, root / SEQUENCE_ORIGIN
    inputs = []
    protocols, manifests = [], []
    for run in (origin, sequence_origin):
        protocol = read(run / "protocol.json")
        require(digest(run / "protocol.json") == read(run / "protocol_lock.json")["sha256"], "Origin protocol seal changed")
        manifest = read(run / "supplemental_manifest.json")
        require(digest(run / "dataset.json") == protocol["dataset_sha256"] and digest(run / "data.npz") == protocol["cache_sha256"], "Base input linkage changed")
        require(digest(run / "supplemental_manifest.json") == protocol["supplemental_training"]["manifest_sha256"]
            and digest(run / "supplemental_training.npz") == protocol["supplemental_training"]["cache_sha256"] == manifest["cache_sha256"], "Ordinary linkage changed")
        require(manifest["personal_reference_used"] is False, "Ordinary restoration cannot use personal references")
        for rel in CORE:
            require(digest(run / "source_snapshot" / rel) == protocol["source_hashes"][rel], "Frozen source changed: " + rel)
        inputs.extend(reference(run / name, root) for name in ("protocol.json", "protocol_lock.json", "dataset.json", "data.npz", "supplemental_manifest.json", "supplemental_training.npz"))
        protocols.append(protocol);manifests.append(manifest)
    old, sequence = protocols
    require(old["cache_sha256"] == sequence["cache_sha256"] and old["dataset_sha256"] == sequence["dataset_sha256"]
        and old["supplemental_training"]["cache_sha256"] == sequence["supplemental_training"]["cache_sha256"], "019 and014 cache bytes differ; do not substitute inputs")
    require(all(old["source_hashes"][rel] == sequence["source_hashes"][rel] for rel in CORE), "019/014 feature or restoration semantics differ")
    proof = manifests[1]["sequence_evidence"]
    require(proof["dtype"] == "float32" and proof["shape"] == [manifests[1]["window_count"], 36, 16]
        and proof["feature_parity"]["bitwise_equal"] is True and proof["feature_parity"]["feature_dimension"] == 596
        and proof["feature_parity"]["maximum_absolute_difference"] == 0.0, "Missing frozen sequence parity evidence")
    fitted_path = origin / "audits/structured_fitted_identity.json"
    fitted = read(fitted_path)
    require(fitted["protocol_sha256"] == digest(origin / "protocol.json") and fitted["all_inputs_unchanged"] is True
        and type(fitted["protected_files_unchanged"]) is int and fitted["protected_files_unchanged"] == 1499
        and len(fitted["train_scopes"]) == 3, "Invalid frozen TRAIN scope audit")
    protected = _verify_protected(root, origin)
    inputs.extend(reference(path, root) for path in (fitted_path, origin / "protected_before.json",
        root / "research/models/checkpoints/research_baseline.json", root / "src/validated_checkpoints.json"))
    config = {"schema_version": 1, "version": VERSION, "frozen_at": now(), "origin_run": ORIGIN, "sequence_origin": SEQUENCE_ORIGIN,
        "origin_protocol_sha256": digest(origin / "protocol.json"), "inputs": inputs,
        "script_sha256": digest(Path(__file__)), "environment": environment(), "partitions": old["partitions"],
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "dirty_state": subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).splitlines(),
        "origin_core_sources": {rel: old["source_hashes"][rel] for rel in CORE},
        "representations": {"sensor_window": [36, 16], "feature_vector": [596]}, "dtype": "float32", "modes": list(MODES),
        "ordinary_policy": "include_ordinary_reserved_training_records_v1", "report_path": str(report.relative_to(root)),
        "scope": {"fits": 0, "predictions": 0, "performance_evaluation": False, "outer_validation_test_numeric_rows": False,
            "all_identity_metadata_only": True, "protected_files": len(protected), "exact_equality_not_similarity": True,
            "new_raw_read_for_features": False, "automatic_label_or_exclusion_changes": False},
        "equality_rule": "Same float32 row dtype+shape+C bytes; verify actual bytes after SHA bucket; signed_zero_normalized alone maps all numeric zeros to +0; no rounding/tolerance",
        "denominators": "per outerTRAIN with ordinary; report input equivalence classes, duplicate rows/excess, same-label and conflicting-label groups; cross-split sums are repeated observations",
        "union_scope": "Unique (base|ordinary,cache_row_index) among three TRAIN scopes; already-decoded TRAIN rows only; separately label as union, not a fourth training split"}
    output.mkdir(parents=True)
    snapshot = output / "source_snapshot/exact_input_collision_audit.py"
    snapshot.parent.mkdir()
    with snapshot.open("xb") as stream:stream.write(Path(__file__).read_bytes())
    snapshot.chmod(0o444)
    write(output / "diagnostic_config.json", config)
    write(output / "diagnostic_lock.json", {"config_sha256": digest(output / "diagnostic_config.json"),
        "script_sha256": digest(snapshot), "frozen_at": config["frozen_at"], "statistics_before_freeze": False})
    print(json.dumps({"output": str(output), "config_sha256": digest(output / "diagnostic_config.json"), "script_sha256": digest(snapshot)}))


def execute(root, output):
    config, lock = read(output / "diagnostic_config.json"), read(output / "diagnostic_lock.json")
    snapshot = output / "source_snapshot/exact_input_collision_audit.py"
    require(digest(output / "diagnostic_config.json") == lock["config_sha256"] and digest(snapshot) == lock["script_sha256"] == config["script_sha256"], "Diagnostic freeze changed")
    if Path(__file__).resolve() != snapshot.resolve():
        os.execv(sys.executable, [sys.executable, "-B", str(snapshot), "run", "--root", str(root), "--output", str(output)])
    require(environment() == config["environment"] and not (output / "execution_start.json").exists(), "Environment changed or execution already exists")
    origin, sequence_origin = root / config["origin_run"], root / config["sequence_origin"]
    def verify():
        for item in config["inputs"]:require(reference(root / item["path"], root) == item, "Diagnostic input changed: " + item["path"])
        for run in (origin, sequence_origin):
            for rel, expected in config["origin_core_sources"].items():require(digest(run / "source_snapshot" / rel) == expected, "Frozen core source changed")
        return _verify_protected(root, origin)
    protected = verify()
    sys.path.insert(0, str(origin / "source_snapshot"))
    from src.posture_core import CLASS_NAMES, SENSOR_COLUMNS, normalize_text
    from src.window_pipeline import extract_window_features
    from experiments.iterative_improvement.record_training import append_record_training
    require(Path(append_record_training.__code__.co_filename).resolve() == (origin / "source_snapshot/experiments/iterative_improvement/record_training.py").resolve(), "Restoration code is not frozen")
    require(read(sequence_origin / "supplemental_manifest.json")["sequence_evidence"]["sensor_order"] == SENSOR_COLUMNS, "Sensor order differs")
    write(output / "execution_start.json", {"started_at": now(), "config_sha256": lock["config_sha256"], "script_sha256": lock["script_sha256"], "inputs_and_protected_verified": True})
    cache = {}
    for kind, filename in (("base", "data.npz"), ("ordinary", "supplemental_training.npz")):
        with np.load(origin / filename, allow_pickle=False) as archive:
            groups, records = archive["groups"], archive["record_ids"]
        require(groups.ndim == records.ndim == 1 and len(groups) == len(records), "Invalid identity metadata")
        cache[kind] = {"groups": np.asarray([normalize_text(str(v)) for v in groups]),
            "record_ids": np.asarray([normalize_text(str(v)) for v in records]), "path": origin / filename,
            "sequence_path": sequence_origin / filename}
    require(not set(cache["base"]["record_ids"]) & set(cache["ordinary"]["record_ids"]), "Base/ordinary records overlap")
    expected = {r["partition"]:r for r in read(origin / "audits/structured_fitted_identity.json")["train_scopes"]}
    results, unique = [], {}
    for part in config["partitions"]:
        require(part["id"] in expected and len(config["partitions"]) == 3, "Unexpected TRAIN partition")
        people = sorted(normalize_text(v) for v in part["train"])
        require(not set(people) & {normalize_text(v) for key in ("validation", "test") for v in part[key]}, "Outer people overlap")
        chosen = {};origins = []
        for kind, data in cache.items():
            take = np.flatnonzero(np.isin(data["groups"], people))
            selected = {key:selected_rows(data["path"], key, take, len(data["groups"])) for key in ("X", "y", "start_s")}
            selected.update(groups=data["groups"][take], record_ids=data["record_ids"][take], indices=take)
            selected["seq"] = selected_rows(data["sequence_path"], "seq", take, len(data["groups"]))
            for key in ("X", "y", "start_s"):
                old = selected_rows(data["sequence_path"], key, take, len(data["groups"]))
                require(old.dtype == selected[key].dtype and old.tobytes() == selected[key].tobytes(), "Selected019/014 numeric row bytes differ")
            require(selected["seq"].dtype == np.float32 and selected["seq"].shape == (len(take),36,16), "Sequence input schema changed")
            untouched = {key:array_hash(selected[key]) for key in ("seq", "X", "y", "start_s")}
            rebuilt = extract_window_features(selected["seq"])
            require(rebuilt.dtype == selected["X"].dtype and rebuilt.shape == selected["X"].shape and rebuilt.tobytes() == selected["X"].tobytes(), "TRAIN sequence/features fail exact extraction parity")
            require(all(array_hash(selected[key]) == value for key,value in untouched.items()), "Feature parity check mutated an input array")
            chosen[kind] = selected
            origins.extend({"cache":kind,"index":int(i)} for i in take)
        base, ordinary = chosen["base"], chosen["ordinary"]
        X,y,groups,records,restoration = append_record_training(base,ordinary,np.arange(len(base["X"])))
        seq = np.concatenate([base["seq"],ordinary["seq"]])
        starts = np.concatenate([base["start_s"],ordinary["start_s"]])
        hashes = {"training_input_sha256":array_hash(X),"training_labels_sha256":array_hash(y),
            "training_groups_sha256":json_hash(groups.tolist()),"training_records_sha256":json_hash(records.tolist())}
        require(all(expected[part["id"]][k] == v for k,v in hashes.items()), "Rows differ from actual019 TRAIN fitted scope")
        require(base["indices"].tolist() == expected[part["id"]]["base_train_indices"] and ordinary["indices"].tolist() == expected[part["id"]]["ordinary_train_indices"], "TRAIN row indices changed")
        require(len(X)==len(seq)==len(origins) and X.dtype==np.float32 and X.shape==(len(y),596), "Combined input schema differs")
        comparisons={name:{mode:collision_summary(v,y,groups,records,origins,mode) for mode in MODES}
            for name,v in (("sensor_window",seq),("feature_vector",X))}
        row = {"partition":part["id"],"training_windows":len(y),"training_records":len(set(records.tolist())),"training_people":len(set(groups.tolist())),
            "class_counts":np.bincount(y,minlength=7).tolist(),"base_indices":base["indices"].tolist(),"ordinary_indices":ordinary["indices"].tolist(),
            "input_hashes":hashes,"sequence_sha256":array_hash(seq),"start_s_sha256":array_hash(starts),"scope_sha256":json_hash(origins),
            "019_fitted_scope_equal":True,"019_014_cache_bytes_equal":True,"TRAIN_feature_extraction_bytes_equal":True,
            "restored_windows":restoration["added_windows"],"restored_records":restoration["added_records"],"comparisons":comparisons}
        results.append(row)
        for i,where in enumerate(origins):
            key=(where["cache"],where["index"])
            payload=(seq[i].tobytes(),X[i].tobytes(),int(y[i]),str(groups[i]),str(records[i]),float(starts[i]))
            if key in unique:require(unique[key]["payload"]==payload,"Repeated TRAIN origin changed between splits")
            else:unique[key]={"payload":payload,"seq":seq[i].copy(),"X":X[i].copy(),"y":int(y[i]),"group":str(groups[i]),"record":str(records[i]),"origin":where}
    entries=[unique[k] for k in sorted(unique)]
    uy=np.asarray([e["y"] for e in entries],dtype=np.int64);ug=np.asarray([e["group"] for e in entries]);ur=np.asarray([e["record"] for e in entries]);uo=[e["origin"] for e in entries]
    union={name:{mode:collision_summary(np.stack([e[key] for e in entries]),uy,ug,ur,uo,mode) for mode in MODES} for name,key in (("sensor_window","seq"),("feature_vector","X"))}
    verify()
    result={"schema_version":1,"version":VERSION,"started_at":read(output/'execution_start.json')["started_at"],"finished_at":now(),
        "config_sha256":lock["config_sha256"],"script_sha256":lock["script_sha256"],"origin_protocol_sha256":config["origin_protocol_sha256"],
        "executed_frozen_script":True,"class_names":CLASS_NAMES,"partitions":results,"union_of_TRAIN_only":union,
        "denominators":{"window_split_appearances":sum(r['training_windows'] for r in results),"unique_TRAIN_windows":len(entries),
            "record_split_appearances":sum(r['training_records'] for r in results),"unique_TRAIN_records":len(set(ur.tolist())),
            "person_split_appearances":sum(r['training_people'] for r in results),"unique_TRAIN_people":len(set(ug.tolist()))},
        "protected_count":len(protected),"protected_unchanged":True,"inputs_unchanged":True,"fits":0,"predictions":0,"outer_validation_test_numeric_rows_used":False,
        "scope":config["scope"],"automatic_label_or_exclusion_changes":False,
        "limitations":["Exact bytes and signed-zero-only equality do not detect near duplicates, aliases after scaling or semantic labeling mistakes",
            "Repeated TRAIN appearances are not independent observations; union is only de-duplicated scope, never a new split",
            "A conflict does not prove which label is wrong; no accuracy ceiling or model selection is inferred"]}
    write(output/'collision_audit.json',result)
    report=render(result,output,root)
    report_path=root/config['report_path']
    write(report_path,report)
    write(output/'collision_seal.json',{"result_sha256":digest(output/'collision_audit.json'),"report_path":config['report_path'],"report_sha256":digest(report_path),
        "config_sha256":lock['config_sha256'],"script_sha256":lock['script_sha256'],"sealed_at":now()})
    print(json.dumps({"output":str(output),"result_sha256":digest(output/'collision_audit.json'),"report_sha256":digest(report_path),
        "counts":[{'partition':r['partition'],'comparisons':{name:{mode:{key:v for key,v in summary.items() if key!='group_details'} for mode,summary in modes.items()} for name,modes in r['comparisons'].items()}} for r in results]},ensure_ascii=False))


def render(result,output,root):
    relative=os.path.relpath(output,root/'docs/reports')
    d=result['denominators'];total=sum(r['comparisons']['feature_vector']['exact_bytes']['conflict_groups'] for r in result['partitions'])
    lines=['# TRAIN 센서 창·596개 특징의 동일 입력 라벨 충돌 진단','',
        f"세 outer TRAIN의 float32 596개 특징 원 바이트에서 서로 다른 자세 라벨을 가진 충돌 그룹은 합계 {total}개였다. 아래의 창·특징, 원 바이트·부호 없는 영 정규화 결과를 구분한다. 이 합계는 분할별 반복 관측이며 독립 충돌 수나 모델 정확도 상한이 아니다.",'',
        '## 실행 범위와 무결성','',
        f"진단 설정·소스를 실제 입력 통계 계산 전에 봉인하고 별도 프로세스의 동결 스크립트로 실행했다. 실행 {result['started_at']}–{result['finished_at']}, config SHA-256 `{result['config_sha256']}`, source SHA-256 `{result['script_sha256']}`.",'',
        '019의 세 TRAIN과 해당 사람의 일반 정자세 복원만 사용했다. 개인 등록 기록은 제외했다. 014의 시계열 검증을 가진 기존 cache가 019의 base/ordinary cache와 전체 파일 SHA-256까지 같고, 특징 생성·복원 소스도 같음을 확인했다. 각 TRAIN의 seq/X/y/start_s 행을 선택하여 다시 비교하고 frozen extractor(seq)의 특징 바이트가 X와 정확히 같은지 확인했다. 실제019 TRAIN 적합 감사의 행 인덱스·X/y/사람/기록 해시와도 일치했다. 새 원본 CSV 공급이나 특징 재구성을 위한 원본 읽기는 없었다.','',
        '전체 groups/record IDs는 TRAIN 선택을 위한 식별 메타데이터로 읽었다. seq/X/y/start_s 숫자 배열은 selected_rows로 선택 행만 해석했다. ZIP seek와 파일 해시는 다른 구간의 바이트를 읽거나 내부 해제할 수 있으나 해당 숫자 관측값을 배열로 해석·분석하지 않았다. 개발/holdout 점수·예측을 열지 않았고 모델·scaler 적합과 새 예측은 모두 0회다. 보호 파일1499개의 SHA-256·크기·mtime 및 모든 입력이 실행 전후 같았다.','',
        '## 동일성 정의와 분모','',
        '- 원 바이트: float32 dtype, 행 shape, C-order 바이트가 모두 같아야 한다. SHA-256 bucket 이후 실제 바이트 동등성도 확인한다.',
        '- 영 부호 정규화: 별도 복사본에서 수치상0인 원소만 +0으로 바꾼 뒤 같은 검사를 한다. +0/-0 외의 반올림·거리·허용오차·값 변경은 없다.',
        '- 중복 그룹은 같은 입력이2행 이상인 그룹이다. 같은 라벨만 가진 중복과 서로 다른 라벨을 가진 충돌을 분리한다. 중복 행은 그룹에 포함된 모든 행 수, 초과 행은 전체 행−고유 입력 수다.',
        '- 센서 단위는 단일16채널 시점이 아닌 실제 모델 입력인36×16 창이다. 특징 단위는 scaler 이전의596개 frozen float32 값이다.','',
        f"세 분할 합계 {d['window_split_appearances']:,}창·{d['record_split_appearances']}기록·{d['person_split_appearances']}명 관측이며, (cache,row index)로 합집합을 중복 제거하면 {d['unique_TRAIN_windows']:,}창·{d['unique_TRAIN_records']}기록·{d['unique_TRAIN_people']}명이다. 합집합은 이미 선택한 TRAIN 행만 재사용한 설명용 범위이며 네 번째 학습 분할이 아니다.",'',
        '## 분할별 결과','',
        '| 분할 | 표현 | 동일성 | 전체 행 | 고유 입력 | 같은 라벨 중복 그룹 | 충돌 그룹 | 중복 행 | 초과 행 |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in result['partitions']:
        for name,modes in row['comparisons'].items():
            for mode,s in modes.items():lines.append(f"| {row['partition']} | {name} | {mode} | {s['rows']} | {s['unique_inputs']} | {s['same_label_duplicate_groups']} | {s['conflict_groups']} | {s['duplicate_rows']} | {s['excess_duplicate_rows']} |")
    lines+=['','## 고유 TRAIN 행 합집합','', '| 표현 | 동일성 | 전체 행 | 고유 입력 | 같은 라벨 중복 그룹 | 충돌 그룹 | 중복 행 |','| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for name,modes in result['union_of_TRAIN_only'].items():
        for mode,s in modes.items():lines.append(f"| {name} | {mode} | {s['rows']} | {s['unique_inputs']} | {s['same_label_duplicate_groups']} | {s['conflict_groups']} | {s['duplicate_rows']} |")
    lines+=['','## 해석의 한계','',
        '이 검사는 전체 CSV/기록 해시 중복 검사와 별개다. 정확한 입력 충돌이 없더라도 유사한 입력·취득 조건 문제·의미상 라벨 오류·모델의 일반화 한계를 배제하지 못한다. 충돌이 있더라도 어느 라벨이 잘못되었는지 자동 판정하지 않으며 라벨 수정·제외·원본 삭제를 하지 않는다. scaler나 파생 변환 이후의 등가성, 가까운 거리의 입력, 센서의 물리적 정확도는 검증하지 않았다. 성능 평가나 정확도 상한 추정으로 사용하지 않는다.','',
        f"[진단 설정]({relative}/diagnostic_config.json) · [진단 결과]({relative}/collision_audit.json) · [봉인]({relative}/collision_seal.json) · [동결 스크립트]({relative}/source_snapshot/exact_input_collision_audit.py). JSON에는 원 TRAIN 인덱스·scope 해시와 중복 그룹의 라벨 수·사람/기록 가명·원 행 식별자를 기록했다. 원본과 기준/활성 모델은 변경하지 않았다."]
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','run'])
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--report',type=Path,help='New docs/reports destination, fixed during freeze only')
    args=parser.parse_args();root=args.root.resolve();output=args.output.resolve()
    require(args.action == 'freeze' or args.report is None, 'Report destination is fixed by the prior freeze')
    freeze(root,output,args.report) if args.action=='freeze' else execute(root,output)


if __name__=='__main__':main()
