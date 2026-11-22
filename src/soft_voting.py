"""Equal-weight probability experiments on identity-aligned held-out windows."""
from __future__ import annotations

import numpy as np
from .model_catalog import MODEL_NAMES
from .posture_core import CLASS_NAMES


def validate_members(members, families, windows):
    result = tuple(members or ())
    allowed = {f"{family}:{seconds}" for family in families for seconds in windows}
    if result and (len(result) < 2 or len(set(result)) != len(result) or not set(result) <= allowed):
        raise ValueError("Soft voting에는 선택한 모델·창 조합 중 서로 다른 2개 이상이 필요합니다.")
    return result


def aligned_indices(datasets, members, split):
    """Match record and nominal end, never row order or labels alone."""
    seconds = sorted({int(member.split(":")[1]) for member in members})
    maps, reference = {}, None
    for window in seconds:
        data = datasets[window]
        signature = (
            {name: part["subjects"] for name, part in data["partitions"].items()},
            {r["record_id"]: (r["subject"], r["label"], r["sha256"]) for r in data["source_records"]},
        )
        if reference is not None and signature != reference:
            raise ValueError("Soft voting 모델 사이의 데이터 또는 피험자 분할이 다릅니다.")
        reference = signature
        rows = data["meta"].loc[data["masks"][split]].reset_index(drop=True)
        mapping = {}
        for index, row in rows.iterrows():
            key = (str(row.record_id), round(float(row.nominal_end_s), 6))
            if key in mapping:
                raise ValueError("Soft voting 평가 시점이 중복되었습니다.")
            mapping[key] = (index, str(row.subject), int(row.class_id))
        maps[window] = mapping
    keys = sorted(set.intersection(*(set(mapping) for mapping in maps.values())))
    if not keys:
        raise ValueError("Soft voting의 공통 평가 시점이 없습니다. 창 길이를 줄이거나 긴 기록을 선택하세요.")
    identities = []
    for key in keys:
        values = [mapping[key][1:] for mapping in maps.values()]
        if any(value != values[0] for value in values):
            raise ValueError("Soft voting 정답 또는 피험자 정보가 일치하지 않습니다.")
        identities.append({"record_id": key[0], "nominal_end_s": key[1], "subject": values[0][0], "class_id": values[0][1]})
    if {row["class_id"] for row in identities} != set(range(len(CLASS_NAMES))):
        raise ValueError("Soft voting 공통 평가 시점에 모든 자세가 포함되어야 합니다.")
    return {window: np.array([mapping[key][0] for key in keys]) for window, mapping in maps.items()}, identities


def evaluate(datasets, members, probabilities, split, score):
    indices, identities = aligned_indices(datasets, members, split)
    y = np.array([row["class_id"] for row in identities])
    aligned = []
    rows = []
    for member in members:
        family, seconds = member.split(":")
        probability = np.asarray(probabilities[member], dtype=float)[indices[int(seconds)]]
        if (probability.shape != (len(y), len(CLASS_NAMES)) or not np.isfinite(probability).all()
                or (probability < 0).any() or not np.allclose(probability.sum(axis=1), 1)):
            raise ValueError("Soft voting 확률 형식이 올바르지 않습니다.")
        aligned.append(probability)
        rows.append({"family": family, "name": MODEL_NAMES[family], "window_seconds": int(seconds), split: score(y, probability.argmax(axis=1))})
    mean = np.mean(aligned, axis=0)
    rows.append({"family": "equal_soft_voting", "display_label": f"동일 가중 Soft voting · {len(members)}개", split: score(y, mean.argmax(axis=1))})
    evidence = {"identities": identities, "member_probabilities": {member: p.tolist() for member, p in zip(members, aligned)},
                "mean_probabilities": mean.tolist(), "predictions": mean.argmax(axis=1).tolist()}
    return rows, evidence
