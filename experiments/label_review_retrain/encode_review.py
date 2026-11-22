"""Encode the user's 2026-09-30 review of LR001-LR080 into a structured decision file.

The user returned the review as chat text (numbers 1-80 = review IDs LR001-LR080).
This script keeps that text verbatim, adds a hand-written structured reading of it,
validates it against the original review sheet (IDs, source hashes) and writes two
new files next to the sheet without touching the original sheet or data/:

  review_decisions_2026-09-30.json  structured decisions + raw text (experiment input)
  review_sheet_2026-09-30.csv       dated copy of review_sheet.csv with the human columns filled

Interval times are seconds from the first sample of the original CSV, i.e. the
x-axis of plots/LRxxx.png. ``None`` as an interval end means "to the end".

Revision 2 (the user's re-coding of 41-80, received the same evening) writes
``review_decisions_2026-09-30_rev2.json`` / ``review_sheet_2026-09-30_rev2.csv``.

Usage: .venv/bin/python -B -m experiments.label_review_retrain.encode_review [--revision rev1|rev2]
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
REVIEW_DIR = ROOT / "research/results/2026-09-28_130815_label_review_all_wrong"
SHEET_INITIAL_SHA256 = "059b5ff0ed642e6494433019b1763ef52aa7d16274973096946c231db9db8455"
DATE = "2026-09-30"
VERSION = "label_review_decisions_v1"

# Legend the user gave for items 41-80, verbatim.
CODE_LEGEND_41_80 = {
    "1": "삭제",
    "2": "보류",
    "3": "예측값과 다르지만 센서 신호 자체에는 명백한 고장이 보이지 않아, 자세 수행 오류 또는 개인 내 변동 가능성인 것 같다.",
}

# Raw text per item, verbatim from the chat message (only the leading "N." / "N :" removed).
RAW = {
    1: "제거 센서 이상 7.5초까진 정상", 2: "3초까진 센서이상 3초 이후부턴 보류 요망",
    3: "2.5초 이전 15초와 18초 사이 센서 이상", 4: "자세 라벨링 이상 혹은 자세 이상",
    5: "5초까지 센서 이상 이후 라벨링 이상 혹은 자세 이상", 6: "라벨링 이상 혹은 정상 보류 요망",
    7: "센서 이상 폐기", **{n: "보류" for n in range(8, 24)}, 24: "폐기 센서 이상",
    **{n: "보류" for n in range(25, 30)}, 30: "폐기 센서 이상", 31: "보류", 32: "보류",
    33: "보류 모델 예측 실패", 34: "보류", 35: "폐기 센서 이상", 36: "보류 모델 예측 실패",
    **{n: "보류" for n in range(37, 41)},
    41: "3", 42: "3", 43: "1", 44: "1", 45: "1", 46: "1", 47: "1 ( 7.5부터 13까지는 정상)",
    48: "2 ( 5 부터는 정상)", 49: "2 (15부터는 정상)", 50: "1", 51: "3", 52: "2", 53: "1", 54: "1",
    55: "1", 56: "1", 57: "1", 58: "1", 59: "1", 60: "3", 61: "1", 62: "1", 63: "1 ( 15부터는 정상)",
    64: "3", 65: "2 ( 7부터는 정상)",
    66: "1 ( 3으로 보이지만 오른쪽 꼬기와 기울임의 혼동을 주는 데이터가 될 것 같아 삭제)",
    67: "1 ( 3으로 보이지만 오른쪽 꼬기와 기울임의 혼동을 주는 데이터가 될 것 같아 삭제)",
    68: "1", 69: "1", 70: "3", 71: "1", 72: "2", 73: "1 ( 15부터는 정상)", 74: "3 ( 자세문제같음)",
    75: "3", 76: "1", 77: "1 ( 15부터는 정상)", 78: "1", 79: "1 ( 0부터 10까진 정상)", 80: "1",
}

DELETE, HOLD, MISMATCH = "delete", "hold", "posture_mismatch"


def item(decision, reason, normal=(), abnormal=(), interpretation=""):
    return {"decision": decision, "reason": reason, "normal_intervals": [list(v) for v in normal],
            "abnormal_intervals": [list(v) for v in abnormal], "interpretation": interpretation}


# Structured reading. Items 1-40 are free text; 41-80 use the 1/2/3 legend.
STRUCTURED = {
    1: item(DELETE, "sensor_issue", normal=[(0.0, 7.5)], interpretation="제거(삭제). 센서 이상, 0–7.5초는 정상."),
    2: item(HOLD, "sensor_issue_then_hold", abnormal=[(0.0, 3.0)],
            interpretation="0–3초 센서 이상, 이후 보류."),
    3: item(HOLD, "sensor_issue_segments", abnormal=[(0.0, 2.5), (15.0, 18.0)],
            interpretation="0–2.5초·15–18초 센서 이상. 기록 조치 미기재 → 기록은 보류로 처리."),
    4: item(MISMATCH, "label_or_posture_issue", interpretation="라벨링 이상 또는 자세 이상 (41–80의 3번과 같은 범주로 처리)."),
    5: item(MISMATCH, "sensor_issue_then_label_or_posture_issue", abnormal=[(0.0, 5.0)],
            interpretation="0–5초 센서 이상, 이후 라벨링 이상 또는 자세 이상."),
    6: item(HOLD, "possible_label_issue", interpretation="라벨링 이상 또는 정상, 보류."),
    7: item(DELETE, "sensor_issue", interpretation="센서 이상, 폐기."),
    24: item(DELETE, "sensor_issue", interpretation="폐기, 센서 이상."),
    30: item(DELETE, "sensor_issue", interpretation="폐기, 센서 이상."),
    33: item(HOLD, "model_prediction_failure", interpretation="보류, 모델 예측 실패."),
    35: item(DELETE, "sensor_issue", interpretation="폐기, 센서 이상."),
    36: item(HOLD, "model_prediction_failure", interpretation="보류, 모델 예측 실패."),
    47: item(DELETE, "unspecified", normal=[(7.5, 13.0)], interpretation="삭제. 7.5–13초는 정상."),
    48: item(HOLD, "unspecified", normal=[(5.0, None)], interpretation="보류. 5초부터 정상(0–5초 비정상)."),
    49: item(HOLD, "unspecified", normal=[(15.0, None)], interpretation="보류. 15초부터 정상(0–15초 비정상)."),
    63: item(DELETE, "unspecified", normal=[(15.0, None)], interpretation="삭제. 15초부터 정상."),
    65: item(HOLD, "unspecified", normal=[(7.0, None)], interpretation="보류. 7초부터 정상(0–7초 비정상)."),
    66: item(DELETE, "right_cross_vs_right_lean_confusion",
             interpretation="3번(센서 고장 없음)으로 보이나 오른쪽 다리 꼬기·오른쪽 기울임 혼동을 줄 수 있어 삭제."),
    67: item(DELETE, "right_cross_vs_right_lean_confusion",
             interpretation="3번(센서 고장 없음)으로 보이나 오른쪽 다리 꼬기·오른쪽 기울임 혼동을 줄 수 있어 삭제."),
    73: item(DELETE, "unspecified", normal=[(15.0, None)], interpretation="삭제. 15초부터 정상."),
    74: item(MISMATCH, "posture_execution_or_within_person_variation", interpretation="3번, 자세 문제로 보임."),
    77: item(DELETE, "unspecified", normal=[(15.0, None)], interpretation="삭제. 15초부터 정상."),
    79: item(DELETE, "unspecified", normal=[(0.0, 10.0)], interpretation="삭제. 0–10초는 정상."),
}


def complete(raw: dict, structured: dict) -> dict:
    """Fill plain items: 1-40 '보류'; 41-80 bare codes 1/2/3. Notes need explicit readings."""
    structured = dict(structured)
    for number in range(1, 81):
        if number in structured:
            continue
        text = raw[number]
        if number <= 40:
            if text != "보류":
                raise AssertionError(f"item {number} needs an explicit reading")
            structured[number] = item(HOLD, "unspecified", interpretation="보류.")
            continue
        code = text.split()[0]
        if text != code:
            raise AssertionError(f"item {number} has a note and needs an explicit reading")
        structured[number] = {
            "1": item(DELETE, "unspecified", interpretation="삭제."),
            "2": item(HOLD, "unspecified", interpretation="보류."),
            "3": item(MISMATCH, "posture_execution_or_within_person_variation",
                      interpretation="센서 고장 없음, 자세 수행 오류 또는 개인 내 변동 가능성."),
        }[code]
    return structured


STRUCTURED = complete(RAW, STRUCTURED)

# Revision 2 (same day, 21:27): items 1-40 unchanged; 41-80 re-coded. Notes the user
# dropped (47, 48, 49, 63, 65, 66, 67, 77) are applied without intervals, as written.
RAW_REV2 = {**{n: RAW[n] for n in range(1, 41)},
    41: "3", 42: "3", 43: "2", 44: "2", 45: "2", 46: "2", 47: "1", 48: "2", 49: "2", 50: "2",
    51: "3", 52: "2", 53: "1", 54: "2", 55: "2", 56: "1", 57: "2", 58: "1", 59: "2", 60: "3",
    61: "2", 62: "1", 63: "1", 64: "3", 65: "2", 66: "2", 67: "2", 68: "1", 69: "2", 70: "3",
    71: "2", 72: "2", 73: "1 ( 15부터는 정상)", 74: "3 ( 자세문제같음)", 75: "3", 76: "1", 77: "2",
    78: "2", 79: "1 ( 0부터 10까진 정상)", 80: "1",
}
STRUCTURED_REV2 = complete(RAW_REV2, {
    **{n: STRUCTURED[n] for n in range(1, 41)},
    73: STRUCTURED[73], 74: STRUCTURED[74], 79: STRUCTURED[79],
})
REVISIONS = {
    "rev1": {"suffix": DATE, "raw": RAW, "structured": STRUCTURED,
             "source": "사용자 채팅 메시지(2026-09-30 19:50). 번호 1–80 = review_id LR001–LR080."},
    "rev2": {"suffix": f"{DATE}_rev2", "raw": RAW_REV2, "structured": STRUCTURED_REV2, "supersedes": "rev1",
             "source": ("사용자 채팅 메시지(2026-09-30 21:27). 1–40은 rev1과 같고 41–80을 다시 판정. "
                        "번호 1–80 = review_id LR001–LR080.")},
}

VERDICT = {"sensor_issue": "recording_issue", "model_prediction_failure": "model_error"}
ACTION = {DELETE: "exclusion_candidate", HOLD: "keep", MISMATCH: ""}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", choices=sorted(REVISIONS), default="rev1")
    revision = REVISIONS[parser.parse_args().revision]
    raw, structured, suffix = revision["raw"], revision["structured"], revision["suffix"]
    sheet_path = REVIEW_DIR / "review_sheet.csv"
    if sha256(sheet_path) != SHEET_INITIAL_SHA256:
        raise SystemExit("review_sheet.csv differs from the hash recorded at generation")
    run = json.loads((REVIEW_DIR / "run.json").read_text(encoding="utf-8"))
    if run["review_sheet_initial_sha256"] != SHEET_INITIAL_SHA256:
        raise SystemExit("run.json sheet hash differs")
    with sheet_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    by_id = {row["review_id"]: row for row in rows}
    if sorted(by_id) != [f"LR{n:03d}" for n in range(1, 81)]:
        raise SystemExit("review sheet must contain exactly LR001-LR080")
    # Sanity anchor for the numbering: the user's notes on 66/67 name right cross vs right lean.
    for number in (66, 67):
        row = by_id[f"LR{number:03d}"]
        if (row["label"], row["majority_prediction"]) != ("오른쪽 다리 꼬기", "오른쪽 기울임"):
            raise SystemExit("item numbering does not match review IDs")

    records = []
    for number in range(1, 81):
        review_id = f"LR{number:03d}"
        row = by_id[review_id]
        records.append({"item": number, "review_id": review_id, "record_id": row["record_id"],
                         "source_file": row["source_file"], "source_sha256": row["source_sha256"],
                         "subject": row["subject"], "label": row["label"],
                         "majority_prediction": row["majority_prediction"], "raw_text": raw[number],
                         **structured[number]})
    counts = {name: sum(r["decision"] == name for r in records) for name in (DELETE, HOLD, MISMATCH)}
    payload = {
        "schema_version": 1, "version": VERSION,
        "created_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(), "timezone": "Asia/Seoul",
        "reviewer": "gto (프로젝트 사용자, 채팅으로 전달)", "review_received_date": DATE,
        "review_performed_date": None,
        "source": revision["source"],
        **({"supersedes": revision["supersedes"]} if "supersedes" in revision else {}),
        "review_sheet": {"path": str(sheet_path.relative_to(ROOT)), "sha256": SHEET_INITIAL_SHA256},
        "code_legend_41_80": CODE_LEGEND_41_80,
        "decision_values": {DELETE: "삭제·폐기·제거", HOLD: "보류(학습·평가에 유지)",
                            MISMATCH: "라벨/자세 불일치(41–80의 3번; 1–40의 4·5번)"},
        "interval_convention": "원본 CSV 첫 샘플 기준 경과 초(plots/LRxxx.png 가로축). 끝이 null이면 기록 끝까지.",
        "counts": counts,
        "records": records,
        "notes": [
            "검토자는 그림에서 모델 예측을 보고 판정했다(비맹검).",
            "검토 대상은 ET 개발 예측에서 모든 창이 틀린 기록 80개뿐이다. 맞게 분류된 기록은 검토하지 않았다.",
            "1–40번의 조치가 없는 3번 항목은 보류로 처리했다. 4·5번은 41–80의 3번 범주로 읽었다.",
            "data/의 CSV와 research/models/training_exclusions.json은 변경하지 않는다.",
        ],
    }
    out_json = REVIEW_DIR / f"review_decisions_{suffix}.json"
    with out_json.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

    extra = ["user_item", "user_raw_text", "user_decision", "user_reason", "user_normal_intervals",
             "user_abnormal_intervals"]
    out_csv = REVIEW_DIR / f"review_sheet_{suffix}.csv"
    fields = list(rows[0]) + extra
    with out_csv.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:  # keep the original sheet order
            decision = records[int(row["review_id"][2:]) - 1]
            filled = dict(row)
            filled.update(reviewer="gto", review_date=DATE,
                          verdict=VERDICT.get(decision["reason"], "unclear"),
                          evidence="plots 검토(사용자)", cross_check_source="none",
                          proposed_action=ACTION[decision["decision"]],
                          notes=decision["interpretation"],
                          user_item=decision["item"], user_raw_text=decision["raw_text"],
                          user_decision=decision["decision"], user_reason=decision["reason"],
                          user_normal_intervals=json.dumps(decision["normal_intervals"]),
                          user_abnormal_intervals=json.dumps(decision["abnormal_intervals"]))
            writer.writerow(filled)
    for path in (out_json, out_csv):
        path.chmod(0o444)
    if sha256(sheet_path) != SHEET_INITIAL_SHA256:
        raise SystemExit("original sheet changed")
    print(json.dumps({"counts": counts, "json": str(out_json.relative_to(ROOT)), "json_sha256": sha256(out_json),
                      "csv": str(out_csv.relative_to(ROOT)), "csv_sha256": sha256(out_csv)}, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
