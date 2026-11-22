#!/usr/bin/env python3
"""Post-hoc alert-policy analysis on saved LOSO live-replay predictions (no refit).

Question: under the app's measurement-mode alert rule, how do model choice,
probability smoothing length, the confidence gate and the alert score trade
posture detection against false alarms during correct (upright) sitting?

App rule (``src.app_services.RiskAssessment`` at HEAD): an event is *poor* when
the smoothed argmax is not 정자세 and its smoothed probability >= 0.45; 60 s of
uninterrupted poor events raise the first alert, every non-poor event resets the
timer. Recordings are ~20 s single-posture CSVs, so 60 s alerts cannot be
observed directly; record-level proxies are used instead (see ``never_reset``).

Inputs are the ``loso_live`` rows of spike_trim_study prediction files: each of
the 26 development subjects is predicted by a model that never saw them, replayed
through the app's live buffer (3 s window, 1 s interval). Thresholds labelled
"matched" are chosen leave-one-subject-out (from the other subjects' abnormal
events), so they are not tuned on the subject they are applied to. The
analysis is exploratory: all 26 subjects were already used in earlier analyses.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "research" / "results"
MAIN_RUN = RESULTS / "2026-09-30_093836_spike_trim_study"
GAMMA_RUN = RESULTS / "2026-09-30_112017_spike_trim_study_rbf_gamma0005"
MODELS = {
    "rbf_svm": (MAIN_RUN, "rbf_svm", "RBF-SVM C=1 gamma=scale (active runtime recipe)"),
    "rbf_svm_gamma0005": (GAMMA_RUN, "rbf_svm", "RBF-SVM C=1 gamma=0.0005 (post-hoc candidate)"),
    "logistic_regression": (MAIN_RUN, "logistic_regression", "Logistic Regression (validation-selected)"),
    "random_forest": (MAIN_RUN, "random_forest", "Random Forest (validation-selected)"),
    "extra_trees": (MAIN_RUN, "extra_trees", "Extra Trees (validation-selected)"),
}
CLASS_NAMES = ["정자세", "앞으로 숙이기", "뒤로 기대기", "왼쪽 기울임", "오른쪽 기울임", "왼쪽 다리 꼬기", "오른쪽 다리 꼬기"]
PROB = [f"p{i}" for i in range(7)]
APP_THRESHOLD = 0.45
APP_HISTORY = 3
BOOTSTRAP = 10000
AUC_BOOTSTRAP = 2000
SEED = 20260930


# --------------------------------------------------------------------------- pure helpers (unit tested)

def smooth(probability: np.ndarray, record_ids: np.ndarray, history: int) -> np.ndarray:
    """Running mean of the last ``history`` rows, restarted at every record (rows contiguous, time ordered)."""
    probability = np.asarray(probability, dtype=float)
    record_ids = np.asarray(record_ids)
    if history < 1 or probability.ndim != 2 or len(probability) != len(record_ids):
        raise ValueError("invalid smoothing input")
    out = np.empty_like(probability)
    for record in dict.fromkeys(record_ids):
        index = np.flatnonzero(record_ids == record)
        if np.any(np.diff(index) != 1):
            raise ValueError(f"rows of {record} are not contiguous")
        cumulative = np.cumsum(np.vstack([np.zeros((1, probability.shape[1])), probability[index]]), axis=0)
        ends = np.arange(1, len(index) + 1)
        starts = np.maximum(0, ends - history)
        out[index] = (cumulative[ends] - cumulative[starts]) / (ends - starts)[:, None]
    return out


def alert_score(smoothed: np.ndarray, rule: str) -> np.ndarray:
    """Score compared with a threshold; an event is poor when score >= threshold.

    ``current``: app rule, the top probability if the top class is not 정자세, else 0.
    ``not_upright``: 1 - P(정자세), i.e. the binary question the alert asks.
    """
    smoothed = np.asarray(smoothed, dtype=float)
    if rule == "current":
        return np.where(smoothed.argmax(axis=1) != 0, smoothed.max(axis=1), 0.0)
    if rule == "not_upright":
        return 1.0 - smoothed[:, 0]
    raise ValueError(f"unknown rule {rule!r}")


def never_reset(poor: np.ndarray, tolerance: int = 1) -> bool:
    """True when an alert timer started at the first event would run through the whole record.

    ``tolerance`` = number of consecutive non-poor events that resets the timer
    (1 = app rule: any non-poor event resets).
    """
    poor = np.asarray(poor, dtype=bool)
    if tolerance < 1:
        raise ValueError("tolerance must be >= 1")
    if not len(poor) or not poor[0]:
        return False
    gap = 0
    for value in poor:
        gap = 0 if value else gap + 1
        if gap >= tolerance:
            return False
    return True


def loso_thresholds(score: np.ndarray, abnormal: np.ndarray, subject: np.ndarray, target: float) -> np.ndarray:
    """Per-event threshold giving ~``target`` detection on the *other* subjects' abnormal events."""
    score, abnormal, subject = np.asarray(score, float), np.asarray(abnormal, bool), np.asarray(subject)
    if not 0 < target < 1:
        raise ValueError("target must be in (0, 1)")
    threshold = np.empty(len(score))
    for person in np.unique(subject):
        others = abnormal & (subject != person)
        threshold[subject == person] = np.quantile(score[others], 1 - target)
    return threshold


def subject_counts(frame: pd.DataFrame, poor: np.ndarray, tolerance: int) -> pd.DataFrame:
    """Per-subject event and record counts used by pooled rates and the subject bootstrap."""
    work = frame.assign(poor=np.asarray(poor, dtype=bool), upright=frame.y.to_numpy() == 0)
    records = work.groupby("record_id", sort=False).agg(subject=("subject", "first"), upright=("upright", "first"))
    records["never_reset"] = [never_reset(work.poor.to_numpy()[index], tolerance)
                              for index in work.groupby("record_id", sort=False).indices.values()]
    event = work.groupby("subject").apply(lambda g: pd.Series({
        "upright_events": int(g.upright.sum()), "upright_poor": int((g.poor & g.upright).sum()),
        "abnormal_events": int((~g.upright).sum()), "abnormal_poor": int((g.poor & ~g.upright).sum())}),
        include_groups=False)
    record = records.groupby("subject").apply(lambda g: pd.Series({
        "upright_records": int(g.upright.sum()), "upright_never_reset": int((g.never_reset & g.upright).sum()),
        "abnormal_records": int((~g.upright).sum()), "abnormal_never_reset": int((g.never_reset & ~g.upright).sum())}),
        include_groups=False)
    return event.join(record).astype(int)


RATES = {  # numerator, denominator
    "detection": ("abnormal_poor", "abnormal_events"),
    "upright_poor": ("upright_poor", "upright_events"),
    "upright_record_never_reset": ("upright_never_reset", "upright_records"),
    "abnormal_record_never_reset": ("abnormal_never_reset", "abnormal_records"),
}


def pooled(counts: pd.DataFrame) -> dict[str, float]:
    total = counts.sum()
    return {name: float(total[a] / total[b]) for name, (a, b) in RATES.items()}


def bootstrap_difference(a: pd.DataFrame, b: pd.DataFrame, replicates: int, seed: int) -> dict:
    """Subject-level bootstrap of pooled-rate differences (b - a); both frames indexed by subject."""
    subjects = sorted(set(a.index) & set(b.index))
    columns = list(RATES_COLUMNS)
    a, b = a.loc[subjects, columns].to_numpy(float), b.loc[subjects, columns].to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(subjects), size=(replicates, len(subjects)))
    sa, sb = a[draws].sum(axis=1), b[draws].sum(axis=1)
    result = {}
    for name, (num, den) in RATES.items():
        i, j = columns.index(num), columns.index(den)
        with np.errstate(invalid="ignore", divide="ignore"):
            diff = sb[:, i] / sb[:, j] - sa[:, i] / sa[:, j]
        observed = b[:, i].sum() / b[:, j].sum() - a[:, i].sum() / a[:, j].sum()
        diff = diff[np.isfinite(diff)]
        result[name] = {"observed": float(observed), "ci95": [float(np.percentile(diff, 2.5)),
                                                             float(np.percentile(diff, 97.5))]}
    result["subjects"] = len(subjects)
    return result


RATES_COLUMNS = ("upright_events", "upright_poor", "abnormal_events", "abnormal_poor",
                 "upright_records", "upright_never_reset", "abnormal_records", "abnormal_never_reset")


# --------------------------------------------------------------------------- analysis

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()


def load_live(run: Path, family: str) -> pd.DataFrame:
    frame = pd.read_csv(run / "predictions" / f"baseline__{family}__live.csv.gz")
    frame = frame[frame.scope == "loso_live"].reset_index(drop=True)
    if not frame.groupby("record_id", sort=False).end_s.apply(lambda s: bool(np.all(np.diff(s) > 0))).all():
        raise ValueError("live rows are not time ordered")
    return frame


def auc_with_ci(score, abnormal, subject, rng) -> dict:
    from sklearn.metrics import roc_auc_score
    subjects = np.unique(subject)
    index = {person: np.flatnonzero(subject == person) for person in subjects}
    values = []
    for _ in range(AUC_BOOTSTRAP):
        rows = np.concatenate([index[p] for p in rng.choice(subjects, len(subjects))])
        values.append(roc_auc_score(abnormal[rows], score[rows]))
    return {"auc": float(roc_auc_score(abnormal, score)),
            "ci95": [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]}


def upright_structure(run: Path, family: str) -> dict:
    windows = pd.read_csv(run / "predictions" / f"baseline__{family}__windows.csv.gz")
    windows = windows[windows.scope == "loso_window"]
    upright = windows[windows.y == 0]
    confusion = pd.crosstab(windows.y, windows.pred).reindex(index=range(7), columns=range(7), fill_value=0)
    record_recall = upright.groupby("record_id").apply(lambda g: float((g.pred == 0).mean()), include_groups=False)
    subject_recall = upright.groupby("subject").apply(lambda g: float((g.pred == 0).mean()), include_groups=False)
    errors = upright[upright.pred != 0].groupby("subject").size().sort_values(ascending=False)
    return {
        "upright_windows": int(len(upright)), "upright_records": int(len(record_recall)),
        "upright_predicted_as": dict(zip(CLASS_NAMES, confusion.loc[0].astype(int).tolist())),
        "predicted_upright_from": dict(zip(CLASS_NAMES[1:], confusion[0].astype(int).tolist()[1:])),
        "records_all_windows_wrong": int((record_recall == 0).sum()),
        "records_recall_at_least_80pct": int((record_recall >= 0.8).sum()),
        "subject_recall_quantiles": dict(zip(("min", "q25", "median", "q75", "max"),
                                             np.percentile(subject_recall, [0, 25, 50, 75, 100]).round(4).tolist())),
        "subjects_recall_below_20pct": int((subject_recall < 0.2).sum()), "subjects": int(len(subject_recall)),
        "top5_subject_share_of_upright_errors": float(errors.head(5).sum() / errors.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="new empty run directory")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = datetime.now().astimezone().isoformat()
    rng = np.random.default_rng(SEED)
    inputs, policies, counts, models = {}, [], {}, {}

    for key, (run, family, description) in MODELS.items():
        path = run / "predictions" / f"baseline__{family}__live.csv.gz"
        inputs[key] = {"file": str(path.relative_to(ROOT)), "sha256": sha256(path), "description": description,
                       "task": str((run / "tasks" / f"baseline__{family}.json").relative_to(ROOT))}
        frame = load_live(run, family)
        probability = frame[PROB].to_numpy(float)
        record_ids = frame.record_id.to_numpy(str)
        subject = frame.subject.to_numpy(str)
        abnormal = frame.y.to_numpy() != 0
        smoothed_app = smooth(probability, record_ids, APP_HISTORY)
        confidence = smoothed_app.max(axis=1)
        models[key] = {
            "description": description, "events": int(len(frame)), "records": int(frame.record_id.nunique()),
            "subjects": int(frame.subject.nunique()),
            "median_smoothed_top_probability": float(np.median(confidence)),
            "abnormal_top_non_upright_below_gate": float(((smoothed_app.argmax(1) != 0) & (confidence < APP_THRESHOLD))[abnormal].mean()),
            "auc_upright_vs_abnormal": {rule: auc_with_ci(alert_score(smoothed_app, rule), abnormal, subject, rng)
                                        for rule in ("current", "not_upright")},
            "upright_structure": upright_structure(run, family) if key != "rbf_svm_gamma0005" else None,
        }

        def add(name, history, rule, threshold, tolerance, kind):
            smoothed = smoothed_app if history == APP_HISTORY else smooth(probability, record_ids, history)
            score = alert_score(smoothed, rule)
            if isinstance(threshold, str):
                target = float(threshold.split("_")[1]) / 100
                poor = score >= loso_thresholds(score, abnormal, subject, target)
            else:
                poor = score >= threshold
            table = subject_counts(frame, poor, tolerance)
            counts[(key, name)] = table
            policies.append({"model": key, "policy": name, "kind": kind, "history": history, "rule": rule,
                             "threshold": threshold, "tolerance": tolerance, **pooled(table)})

        for history in (1, 3, 5, 8):
            add(f"app_N{history}", history, "current", APP_THRESHOLD, 1, "fixed" if history != 3 else "app")
        add("app_N3_tol3", 3, "current", APP_THRESHOLD, 3, "fixed")
        add("app_N3_t0.60", 3, "current", 0.60, 1, "fixed")
        for target in (80, 70):
            for rule in ("current", "not_upright"):
                add(f"matched{target}_{rule}", 3, rule, f"matched_{target}", 1, "loso_matched")

    comparisons = {}

    def compare(label, a, b):
        comparisons[label] = {"a": list(a), "b": list(b),
                              **bootstrap_difference(counts[a], counts[b], BOOTSTRAP, SEED + len(comparisons))}

    for target in (80, 70):
        compare(f"rbf_svm matched{target}: not_upright - current", ("rbf_svm", f"matched{target}_current"),
                ("rbf_svm", f"matched{target}_not_upright"))
        for other in ("logistic_regression", "random_forest", "extra_trees", "rbf_svm_gamma0005"):
            compare(f"matched{target}_current: {other} - rbf_svm", ("rbf_svm", f"matched{target}_current"),
                    (other, f"matched{target}_current"))
    for other in ("random_forest", "extra_trees"):
        compare(f"app rule (0.45): {other} - rbf_svm", ("rbf_svm", "app_N3"), (other, "app_N3"))
    compare("rbf_svm app: N8 - N3", ("rbf_svm", "app_N3"), ("rbf_svm", "app_N8"))
    compare("rbf_svm app: N1 - N3", ("rbf_svm", "app_N3"), ("rbf_svm", "app_N1"))
    compare("rbf_svm app: tolerance3 - tolerance1", ("rbf_svm", "app_N3"), ("rbf_svm", "app_N3_tol3"))
    compare("rbf_svm app: threshold0.60 - 0.45", ("rbf_svm", "app_N3"), ("rbf_svm", "app_N3_t0.60"))

    table = pd.DataFrame(policies)
    table.to_csv(output / "alert_policy.csv", index=False)

    # Planning values for the continuous-session protocol (provisional; lock before new data).
    frame = load_live(*MODELS["rbf_svm"][:2])
    abnormal = frame.y.to_numpy() != 0
    smoothed_app = smooth(frame[PROB].to_numpy(float), frame.record_id.to_numpy(str), APP_HISTORY)
    app_detection = pooled(counts[("rbf_svm", "app_N3")])["detection"]
    not_upright = alert_score(smoothed_app, "not_upright")
    current = alert_score(smoothed_app, "current")
    planning = {"model": "rbf_svm", "history": APP_HISTORY, "app_rule_detection": app_detection,
                "comparison": "not_upright vs current score at the same pooled detection (in-sample thresholds)",
                "thresholds": {}}
    for label, target in (("match_app_detection", app_detection), ("detection_80", 0.80), ("detection_70", 0.70)):
        t_current = APP_THRESHOLD if label == "match_app_detection" else float(np.quantile(current[abnormal], 1 - target))
        t_not_upright = float(np.quantile(not_upright[abnormal], 1 - target))
        table_a = subject_counts(frame, current >= t_current, 1)
        table_b = subject_counts(frame, not_upright >= t_not_upright, 1)
        per_subject = (table_b.upright_poor / table_b.upright_events) - (table_a.upright_poor / table_a.upright_events)
        mean, sd = float(per_subject.mean()), float(per_subject.std(ddof=1))
        planning["thresholds"][label] = {
            "target_detection": float(target),
            "current": {"threshold": t_current, **pooled(table_a)},
            "not_upright": {"threshold": t_not_upright, **pooled(table_b)},
            "per_subject_upright_poor_difference": {"mean": mean, "sd": sd, "subjects": int(len(per_subject)),
                "subjects_for_80pct_power_normal_approx": (float(((1.96 + 0.84) * sd / abs(mean)) ** 2) if mean else None)}}
    planning["note"] = ("Thresholds pooled over all 26 subjects (in-sample). Per-subject SD comes from ~3 upright "
                        "records per person, so the power estimate is conservative for sessions with minutes of upright sitting.")
    payload = {"question": __doc__.split("\n\n")[1], "app_rule": {"threshold": APP_THRESHOLD, "history": APP_HISTORY,
               "reset": "any non-poor event", "initial_alert_s": 60, "realert_s": 120},
               "definitions": {
                   "detection": "share of abnormal-posture events that are poor",
                   "upright_poor": "share of 정자세 events that are poor (would advance the alert timer)",
                   "upright_record_never_reset": "share of 정자세 records (~17 events) whose timer, started at the "
                                                 "first event, never resets: a sustained false-alarm proxy",
                   "abnormal_record_never_reset": "same for abnormal records: a sustained-detection proxy",
                   "matched": "threshold from the other 25 subjects' abnormal events (leave-one-subject-out)",
                   "bootstrap": f"{BOOTSTRAP} subject resamples of pooled-rate differences, 95% percentile"},
               "models": models, "policies": policies, "comparisons": comparisons, "protocol_planning": planning}
    (output / "alert_policy.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                                              encoding="utf-8")

    def pct(value):
        return f"{100 * value:.1f}"
    lines = ["# Alert-policy post-hoc analysis - generated tables", "",
             "Generated by `experiments/alert_policy/run.py` from saved LOSO live predictions; see run.json.", "",
             "| model | policy | detection | upright poor | upright record never reset | abnormal record never reset |",
             "|---|---|---:|---:|---:|---:|"]
    for row in policies:
        lines.append(f"| {row['model']} | {row['policy']} | {pct(row['detection'])} | {pct(row['upright_poor'])} | "
                     f"{pct(row['upright_record_never_reset'])} | {pct(row['abnormal_record_never_reset'])} |")
    lines += ["", "| comparison (b - a) | detection | upright poor | upright record never reset |", "|---|---|---|---|"]
    for label, row in comparisons.items():
        cells = [f"{100 * row[k]['observed']:+.1f} [{100 * row[k]['ci95'][0]:+.1f}, {100 * row[k]['ci95'][1]:+.1f}]"
                 for k in ("detection", "upright_poor", "upright_record_never_reset")]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += ["", "| model | AUC current | AUC 1-P(upright) | median top prob | abnormal gated out |", "|---|---|---|---|---|"]
    for key, row in models.items():
        auc = row["auc_upright_vs_abnormal"]
        lines.append(f"| {key} | {auc['current']['auc']:.3f} [{auc['current']['ci95'][0]:.3f}, {auc['current']['ci95'][1]:.3f}] | "
                     f"{auc['not_upright']['auc']:.3f} [{auc['not_upright']['ci95'][0]:.3f}, {auc['not_upright']['ci95'][1]:.3f}] | "
                     f"{row['median_smoothed_top_probability']:.3f} | {pct(row['abnormal_top_non_upright_below_gate'])} |")
    lines += ["", "## Protocol planning (rbf_svm, same pooled detection, in-sample thresholds)", "",
              "| target | score | threshold | detection | upright poor | upright record never reset | abnormal record never reset |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for label, row in planning["thresholds"].items():
        for score in ("current", "not_upright"):
            cell = row[score]
            lines.append(f"| {label} | {score} | {cell['threshold']:.3f} | {pct(cell['detection'])} | {pct(cell['upright_poor'])} | "
                         f"{pct(cell['upright_record_never_reset'])} | {pct(cell['abnormal_record_never_reset'])} |")
        diff = row["per_subject_upright_poor_difference"]
        n = diff["subjects_for_80pct_power_normal_approx"]
        lines.append(f"| {label} | per-subject upright poor diff | mean {100 * diff['mean']:+.1f} | sd {100 * diff['sd']:.1f} | "
                     f"n80 {'-' if n is None else f'{n:.0f}'} | | |")
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    import scipy, sklearn
    run_record = {
        "purpose": "Post-hoc alert-policy analysis of saved LOSO live predictions (no model fit, no runtime change)",
        "kind": "analysis of CSV-replay predictions (no physical sensor)",
        "started_at": started, "finished_at": datetime.now().astimezone().isoformat(),
        "source_revision": git("rev-parse", "HEAD"), "working_tree_dirty_files": git("status", "--porcelain").splitlines(),
        "command": [sys.executable, *sys.argv], "code_sha256": {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        "inputs": inputs, "seed": SEED, "bootstrap": BOOTSTRAP, "auc_bootstrap": AUC_BOOTSTRAP,
        "exposure": "26 development subjects of the submission split; all were used in earlier analyses",
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__,
                        "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "scipy": scipy.__version__},
        "data_read": False, "models_saved": False,
    }
    (output / "run.json").write_text(json.dumps(run_record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
