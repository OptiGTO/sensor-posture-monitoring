"""Read-only sensor-record audit and a local, human-review report.

Heuristics nominate records for review; they never assign truth or delete data.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .posture_core import (
    CLASS_NAMES, SENSOR_COLUMNS, normalize_text, read_sensor_csv,
    recording_label, standardize_label, sensor_identity,
)
from .window_pipeline import WindowSpec, segment_sensor_frame

RULES = {
    "gap_seconds": 1.0,
    "constant_channel_range": 1e-6,
    "transient_spike": "opposite consecutive jumps > max(12 scaled MAD of differences, 0.25 channel 5-95% range, 0.10 whole-record 5-95% range); initial 0.5s counted separately",
    "step": "adjacent 0.5s bin means change > max(25% channel range, 10% whole-record range) in >= 6 channels",
    "edge_drift": "first/last 2s mean difference > max(60% channel range, 10% whole-record range) in >= 8 channels",
    "repeat_shape_distance": 0.30,
    "repeat_shape_definition": "1 - cosine similarity between centered 16-channel recording means for same subject and label",
    "scope": "All thresholds are review heuristics, not verified sensor fault limits; no automatic clipping, trimming, relabeling or exclusion based on these warnings.",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def flag(record, code, message, weight=1, interval=None):
    item = {"code": code, "message": message, "weight": weight}
    if interval is not None:
        item["interval_s"] = [round(float(v), 3) for v in interval]
    record["flags"].append(item)


def classify_path(path: Path, root: Path):
    parts = {part.casefold() for part in path.relative_to(root).parts[:-1]}
    if "measurements" in parts:
        return "측정 전용", "실제 자세 라벨을 부여하지 않은 측정 기록"
    if "standard" in parts:
        return "빈 좌석 기준", "학습 자세 기록과 별도로 유지"
    if "__macosx" in parts or path.name.startswith("._"):
        return "시스템 파일", "학습 대상 아님"
    if path.name.endswith(".imu.csv"):
        return "IMU 보조 기록", "압력 기록과 별도로 유지"
    sidecar = path.with_suffix(".manifest.json")
    if sidecar.exists():
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("메타데이터가 객체가 아닙니다")
            if (data.get("source_type") == "measurement"
                    or data.get("label_status", "labeled") != "labeled"
                    or data.get("training_role", "include") != "include"):
                return "기존 제외", "원본 메타데이터의 측정 전용·미라벨·학습 제외 설정 유지"
        except (OSError, ValueError) as exc:
            return "메타데이터 오류", str(exc)
    return "학습 후보", ""


def filename_label(path: Path):
    parts = normalize_text(path.stem).split("_")
    if len(parts) < 3:
        return None
    try:
        return standardize_label("".join(parts[1:-1]).strip("'\""))
    except ValueError:
        return None


def inspect_record(path: Path, root: Path) -> tuple[dict, object | None]:
    sha = digest(path)
    relative = path.relative_to(root).as_posix()
    record = {"id": hashlib.sha256((relative + sha).encode()).hexdigest()[:16],
              "file": relative, "filename": normalize_text(path.name),
              "subject": normalize_text(path.parent.name), "sha256": sha,
              "label": "미확인", "raw_labels": [], "flags": [], "blocking": False,
              "eligible": False, "suggested_label": None, "plot": None}
    try:
        frame = read_sensor_csv(path)
    except Exception as exc:
        flag(record, "invalid_numeric_or_time", f"필수 값·시간 검사 실패: {exc}", 100)
        record["blocking"] = True
        return record, None
    record["raw_labels"] = [str(v) for v in frame.get("label", []).unique()] if "label" in frame else []
    try:
        _, record["label"] = recording_label(frame)
    except ValueError as exc:
        record["blocking"] = True
        flag(record, "invalid_label", f"라벨 확인 필요: {exc}", 100)
        record["suggested_label"] = filename_label(path)
        if len(record["raw_labels"]) == 1:
            try:
                record["suggested_label"] = standardize_label(record["raw_labels"][0].strip("'\""))
            except ValueError:
                pass
    label_in_name = filename_label(path)
    if label_in_name and record["label"] in CLASS_NAMES and label_in_name != record["label"]:
        flag(record, "filename_label_conflict", f"파일명 자세({label_in_name})와 내부 라벨({record['label']})이 다름", 20)
    t = frame.elapsed_s.to_numpy(dtype=float)
    t = t - t[0]
    x = frame[SENSOR_COLUMNS].to_numpy(dtype=float)
    dt = np.diff(t)
    record.update(rows=len(t), duration_s=float(t[-1]), median_dt_s=float(np.median(dt)),
                  median_hz=float(1 / np.median(dt)), max_gap_s=float(dt.max()),
                  minimum=float(x.min()), maximum=float(x.max()), sensor_sha256=sensor_identity(frame),
                  fast_intervals=int(np.sum(dt < .01)), mean_map=x.mean(axis=0).tolist(),
                  first_map=x[t <= min(2, t[-1])].mean(axis=0).tolist(),
                  last_map=x[t >= max(0, t[-1] - 2)].mean(axis=0).tolist())
    for index in np.flatnonzero(dt > RULES["gap_seconds"]):
        flag(record, "long_gap", f"1초 초과 공백: {dt[index]:.3f}초", 50, (t[index], t[index + 1]))
        record["blocking"] = True
    if np.std(dt) > max(np.median(dt) * .4, .02):
        flag(record, "irregular_sampling", "샘플링 간격 변동이 큼", 3)
    for i in np.flatnonzero(np.ptp(x, axis=0) < RULES["constant_channel_range"]):
        flag(record, "constant_channel", f"{SENSOR_COLUMNS[i]} 값이 기록 내내 일정함", 10)
    span = np.maximum(np.percentile(x, 95, axis=0) - np.percentile(x, 5, axis=0), 1e-6)
    spatial_floor = max(float(np.percentile(x, 95) - np.percentile(x, 5)) * .1, 1e-6)
    diff = np.diff(x, axis=0)
    mad = np.median(np.abs(diff - np.median(diff, axis=0)), axis=0) * 1.4826
    threshold = np.maximum(np.maximum(12 * mad, .25 * span), spatial_floor)
    spikes = (np.abs(diff[:-1]) > threshold) & (np.abs(diff[1:]) > threshold) & (diff[:-1] * diff[1:] < 0)
    spike_rows = np.flatnonzero(spikes.any(axis=1)) + 1
    record["spike_count"] = int(spikes.sum())
    record["initial_spike_count"] = int(np.sum(t[spike_rows] <= .5))
    spike_rows = spike_rows[t[spike_rows] > .5]
    if len(spike_rows):
        for index in spike_rows[:5]:
            channels = ", ".join(np.asarray(SENSOR_COLUMNS)[spikes[index - 1]])
            flag(record, "transient_spike", f"짧게 튀었다 복귀한 신호 후보: {channels}", 2,
                 (max(0, t[index] - .15), min(t[-1], t[index] + .15)))
    # Half-second bins use only actual observed rows; empty bins are not filled.
    bin_id = np.floor(t / .5).astype(int)
    bins = np.unique(bin_id)
    means = np.stack([x[bin_id == b].mean(axis=0) for b in bins])
    changes = np.abs(np.diff(means, axis=0))
    step_rows = np.flatnonzero(((changes > np.maximum(.25 * span, spatial_floor)).sum(axis=1) >= 6) & (np.diff(bins) == 1))
    for idx in step_rows[:5]:
        when = float(bins[idx + 1] * .5)
        flag(record, "possible_transition", "여러 센서가 함께 급변함: 자세 전환·착석 안정화 여부 확인", 4,
             (max(0, when - .5), min(t[-1], when + .5)))
    drift = np.abs(np.array(record["first_map"]) - np.array(record["last_map"]))
    drift_channels = drift > np.maximum(.6 * span, spatial_floor)
    record["edge_drift_channels"] = int(drift_channels.sum())
    if drift_channels.sum() >= 8:
        flag(record, "edge_drift", "처음·마지막 2초의 센서 패턴 차이가 큼", 5)
    try:
        windows, _ = segment_sensor_frame(frame, WindowSpec(window_seconds=5))
        record["window_count_5s"] = len(windows)
    except ValueError as exc:
        if not record["blocking"]:
            flag(record, "window_failure", str(exc), 50)
        record["blocking"] = True
        record["window_count_5s"] = 0
    record["eligible"] = not record["blocking"]
    return record, frame


def add_cross_record_checks(records):
    identities, groups = defaultdict(list), defaultdict(list)
    for r in records:
        if r.get("sensor_sha256"):
            identities[r["sensor_sha256"]].append(r)
        if r["label"] in CLASS_NAMES and "mean_map" in r:
            groups[(r["subject"], r["label"])].append(r)
    duplicates = []
    for members in identities.values():
        if len(members) > 1:
            duplicates.append([r["id"] for r in members])
            for r in members:
                flag(r, "duplicate_sensor_record", "다른 파일과 시간·센서 시퀀스가 동일함: 중복 및 피험자 ID 확인", 80)
                r["blocking"], r["eligible"] = True, False
    for members in groups.values():
        if len(members) < 2:
            continue
        maps = np.array([r["mean_map"] for r in members])
        centered = maps - maps.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(centered, axis=1)
        unit = centered / np.maximum(norms[:, None], 1e-12)
        distances = np.clip(1 - unit @ unit.T, 0, 2)
        for i, r in enumerate(members):
            distances[i, i] = -1
            j = int(np.argmax(distances[i]))
            r["repeat_peer_id"] = members[j]["id"]
            r["repeat_distance"] = float(distances[i, j])
            r["repeat_peer_map"] = members[j]["mean_map"]
            if norms[i] > 1e-6 and norms[j] > 1e-6 and distances[i, j] > RULES["repeat_shape_distance"]:
                flag(r, "repeat_shape_difference", f"같은 사람·자세의 반복 기록과 평균 패턴 차이가 큼 (거리 {distances[i,j]:.2f})", 8)
    return duplicates


def audit(root: Path):
    root = root.resolve()
    paths = sorted(root.rglob("*.csv"))
    source_paths = paths + sorted(root.rglob("*.manifest.json"))
    before = {str(p): digest(p) for p in source_paths}
    records, excluded, frames = [], [], {}
    for path in paths:
        kind, why = classify_path(path, root)
        if kind not in ("학습 후보", "메타데이터 오류"):
            excluded.append({"file": path.relative_to(root).as_posix(), "category": kind, "reason": why, "sha256": before[str(path)]})
            continue
        r, frame = inspect_record(path, root)
        if kind == "메타데이터 오류":
            flag(r, "manifest_error", why, 100)
            r["blocking"], r["eligible"] = True, False
        records.append(r)
        if frame is not None:
            frames[r["id"]] = frame
    duplicates = add_cross_record_checks(records)
    for r in records:
        r["review_score"] = sum(f["weight"] for f in r["flags"])
        r["tier"] = "필수 확인" if r["blocking"] else "신호 확인" if r["flags"] else "기본 통과"
    # Low-warning sample covers every eligible subject and all available classes.
    normals = sorted((r for r in records if r['eligible']), key=lambda r: (r['review_score'], r["id"]))
    covered_subjects, covered_labels = set(), set()
    for r in normals:
        if r["subject"] not in covered_subjects or r["label"] not in covered_labels:
            r["sample_for_review"] = True
            covered_subjects.add(r["subject"]); covered_labels.add(r["label"])
    records.sort(key=lambda r: (-r["review_score"], r["subject"], r["label"], r["filename"]))
    # A bounded first pass spans participants; all other warnings remain accessible.
    priority_subjects = set()
    for r in records:
        if r['eligible'] and r['flags'] and r['subject'] not in priority_subjects and len(priority_subjects) < 12:
            r['priority_review'] = True
            priority_subjects.add(r['subject'])
    fingerprint = hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest()
    report = {"schema_version": 1, "source_root": str(root.resolve()), "dataset_fingerprint": fingerprint,
              "rules": RULES, "records": records, "excluded": excluded, "duplicate_groups": duplicates,
              "source_hashes": before,
              "summary": {"csv_files": len(paths), "candidate_records": len(records), "excluded_records": len(excluded),
                          "subject_count": len({r['subject'] for r in records}),
                          "eligible_records": sum(r['eligible'] for r in records),
                          "priority_review_count": sum(bool(r.get('priority_review')) for r in records),
                          "sample_review_count": sum(bool(r.get('sample_for_review')) for r in records),
                          "sample_subject_count": len(covered_subjects), "sample_class_count": len(covered_labels),
                          "first_pass_unique_records": sum(r['blocking'] or bool(r.get('priority_review')) or bool(r.get('sample_for_review')) for r in records),
                          "initial_spike_records": sum(r.get('initial_spike_count',0)>0 for r in records),
                          "tiers": dict(Counter(r['tier'] for r in records)),
                          "flag_counts": dict(Counter(code for r in records for code in {f['code'] for f in r['flags']})),
                          "label_counts": dict(Counter(r['label'] for r in records)),
                          "physically_validated": False}}
    return report, frames


def render_plot(record, frame, output):
    import matplotlib.pyplot as plt
    t = frame.elapsed_s.to_numpy(dtype=float); t = t - t[0]
    x = frame[SENSOR_COLUMNS].to_numpy(dtype=float)
    fig = plt.figure(figsize=(13, 8), layout="constrained")
    grid = fig.add_gridspec(3, 4, height_ratios=[1.05, 1.05, 1])
    limits = (float(x.min()), float(x.max()))
    margin = max((limits[1] - limits[0]) * .06, .001)
    for row in range(4):
        ax = fig.add_subplot(grid[row // 2, (row % 2) * 2:(row % 2) * 2 + 2])
        for col in range(4):
            ax.plot(t, x[:, row * 4 + col], label=SENSOR_COLUMNS[row * 4 + col], linewidth=.8)
        for f in record['flags']:
            if "interval_s" in f:
                ax.axvspan(*f['interval_s'], color='#f59e0b', alpha=.15)
        ax.set(xlabel="경과 시간 (초)", ylabel="센서 기록값", ylim=(limits[0] - margin, limits[1] + margin))
        ax.legend(loc="best", ncol=4, fontsize=7)
        ax.grid(alpha=.15)
    maps = [record['first_map'], record['mean_map'], record['last_map'], record.get('repeat_peer_map')]
    valid = [v for v in maps if v is not None]
    lo, hi = float(np.min(valid)), float(np.max(valid))
    for index, (title, values) in enumerate(zip(['처음 2초', '전체 평균', '마지막 2초', '동일 인물·자세의 비교 기록'], maps)):
        ax = fig.add_subplot(grid[2, index])
        ax.set_title(title, fontsize=9)
        if values is None:
            ax.text(.5, .5, '비교 기록 없음', ha='center', va='center'); ax.axis('off'); continue
        matrix = np.array(values).reshape(4, 4)
        im = ax.imshow(matrix, cmap='viridis', vmin=lo, vmax=hi)
        for i in range(4):
            for j in range(4):
                color = 'black' if (matrix[i,j]-lo)/max(hi-lo,1e-9) > .6 else 'white'
                ax.text(j, i, f'{matrix[i,j]:.3f}', ha='center', va='center', color=color, fontsize=8)
        ax.set_xticks(range(4), ['C1','C2','C3','C4']); ax.set_yticks(range(4), ['R1','R2','R3','R4'])
        fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
    fig.suptitle(f"{record['subject']} · {record['label']} · {record['filename']}\n센서 단위·실제 좌우 방향 미확인 / 지도는 파일의 R11~R44 순서 / 음영은 자동 검토 후보", fontsize=11)
    name = record['id'] + '.png'
    fig.savefig(output / name, dpi=110)
    plt.close(fig)
    record['plot'] = 'plots/' + name


def verify_unchanged(report):
    changed = [p for p, sha in report['source_hashes'].items() if not Path(p).is_file() or digest(Path(p)) != sha]
    root=Path(report['source_root'])
    current={str(p) for pattern in ('*.csv','*.manifest.json') for p in root.rglob(pattern)}
    changed.extend(sorted(current-set(report['source_hashes'])))
    report['source_unchanged'] = not changed
    report['changed_sources'] = changed
    if changed:
        raise RuntimeError(f"검사 도중 원본 {len(changed)}개가 바뀌었습니다. 다시 실행하세요.")
