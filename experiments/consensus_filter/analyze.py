"""Audit saved predictions and produce a Korean research report and review lists."""
from __future__ import annotations
import argparse
import csv
import html
import re
import shutil
from pathlib import Path
from experiments.consensus_filter.run import (
    ROOT, SEED, CONDITIONS, MODELS, CLASS_NAMES, read, digest, now, verify,
    protected_snapshot, record_errors, common_records, validate_split, write_json,
)
import numpy as np
import pandas as pd
from src.model_comparison import scores

NAMES = {"random_forest": "Random Forest", "rbf_svm": "RBF SVM", "xgboost": "XGBoost"}


def verify_frozen(out):
    p = read(out / "protocol.json")
    if list(CLASS_NAMES) != p["labels"]:
        raise ValueError("Analysis class order changed")
    if digest(out / "protocol.json") != read(out / "protocol_lock.json")["sha256"]:
        raise ValueError("Protocol lock changed")
    for sec in (2, 3):
        if digest(out / f"data_{sec}s.npz") != p["cohorts"][str(sec)]["cache_sha256"]:
            raise ValueError("Frozen feature cache changed")
    source_changes = []
    for relative, expected in p["source_hashes"].items():
        if digest(out / "source_snapshot" / relative) != expected:
            raise ValueError("Frozen source snapshot changed")
        actual = digest(ROOT / relative)
        if actual != expected:
            source_changes.append({"path": relative, "frozen_sha256": expected, "current_sha256": actual})
    before, after = read(out / "protected_before.json"), protected_snapshot()
    changed = {k: {"before": before.get(k), "after": after.get(k)}
               for k in before.keys() | after.keys() if before.get(k) != after.get(k)}
    # A concurrent task added this policy while our fail-closed workers stopped.
    # Permit only this observed addition; never permit canonical data/model edits.
    allowed = "research/models/training_exclusions.json"
    if any(k != allowed or v["before"] is not None for k, v in changed.items()):
        raise ValueError(f"Unexpected protected file changes: {list(changed)}")
    policy = read(ROOT / allowed) if allowed in changed else None
    write_json(out / "concurrent_workspace_changes.json", {"checked_at": now(),
        "source_changes": source_changes, "protected_inventory_changes": changed,
        "current_external_exclusion_policy": policy,
        "canonical_data_unchanged": True, "all_preexisting_model_files_unchanged": True,
        "interpretation": "A concurrent worktree change was preserved. Remaining fits used the exact pre-change snapshot. Current exclusion policy was not applied retroactively to this locked cohort."})
    return p


def title(condition):
    family, seconds = condition.rsplit("_", 1)
    return NAMES[family] + " · " + seconds.replace("s", "초")


def csv_write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("no_records\n", encoding="utf-8-sig")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def f1_from_cm(cm):
    diag = np.diagonal(cm, axis1=-2, axis2=-1)
    denom = cm.sum(axis=-2) + cm.sum(axis=-1)
    return np.divide(2 * diag, denom, out=np.zeros_like(diag, dtype=float), where=denom != 0).mean(axis=-1)


def paired_bootstrap(y, a, b, groups, draws=2000):
    people = np.unique(groups)
    cms = []
    for pred in (a, b):
        cms.append(np.array([np.bincount(7 * y[groups == s] + pred[groups == s], minlength=49).reshape(7, 7)
                             for s in people]))
    rng = np.random.default_rng(SEED)
    weights = rng.multinomial(len(people), np.full(len(people), 1 / len(people)), size=draws)
    delta = f1_from_cm(np.einsum("ds,sij->dij", weights, cms[1])) - f1_from_cm(np.einsum("ds,sij->dij", weights, cms[0]))
    return {"delta": float(f1_from_cm(cms[1].sum(0)) - f1_from_cm(cms[0].sum(0))),
            "low": float(np.quantile(delta, .025)), "high": float(np.quantile(delta, .975)),
            "draws": draws, "unit": "whole subject; conditional on fitted CV models"}


def review_lists(out, p):
    x = read(out / "full/exclusions.json")
    tables = {c: {r["record_id"]: r for r in rows} for c, rows in x["record_error_tables"].items()}
    source = {r["staged_file"]: r for r in read(out / "training_data/staged_sources.json")}
    audit = {r["relative_path"]: r for r in read(out / "audit.json")["staged_audit"]["records"]}
    nested = {s["id"]: set(read(out / s["id"] / "exclusions.json")["consensus"]) for s in p["outer"]}
    rows = []
    for rid in sorted(source):
        label = tables[CONDITIONS[0]][rid]["label_id"]
        ds = [tables[c][rid]["dominant_prediction"] for c in CONDITIONS]
        r = {"record_id": rid, "source_file": source[rid]["source_file"],
             "source_sha256": source[rid]["source_sha256"], "subject": audit[rid]["subject"],
             "truth_label": CLASS_NAMES[label], "duration_s": audit[rid]["duration_s"],
             "common_all_six": rid in x["consensus"],
             "pattern": "같은 다른 자세로 일치" if len(set(ds)) == 1 and ds[0] != label else "예측 분산 또는 동률",
             "review_status": "미확인; 실제 이상치나 라벨 오류 확정 아님"}
        eligible_scopes = [s["id"] for s in p["outer"] if audit[rid]["subject"] in s["train"]]
        r["nested_training_opportunities"] = len(eligible_scopes)
        r["nested_exclusion_count"] = sum(rid in nested[s] for s in eligible_scopes)
        r["nested_exclusion_rate"] = r["nested_exclusion_count"] / len(eligible_scopes)
        for c in CONDITIONS:
            item = tables[c][rid]
            r[c + "_error_rate"] = item["error_rate"]
            r[c + "_dominant_prediction"] = CLASS_NAMES[item["dominant_prediction"]]
            r[c + "_windows"] = item["windows"]
        for family in MODELS:
            r[family + "_common_2s_3s"] = all(tables[f"{family}_{s}s"][rid]["error_rate"] >= .5 for s in (2, 3))
        for sec in (2, 3):
            r[f"common_three_models_{sec}s"] = all(tables[f"{f}_{sec}s"][rid]["error_rate"] >= .5 for f in MODELS)
        r["mean_error_rate"] = float(np.mean([tables[c][rid]["error_rate"] for c in CONDITIONS]))
        rows.append(r)
    rows.sort(key=lambda r: (-r["common_all_six"], -r["mean_error_rate"], r["record_id"]))
    csv_write(out / "review/all_record_errors.csv", rows)
    csv_write(out / "review/common_all_six.csv", [r for r in rows if r["common_all_six"]])
    for family in MODELS:
        csv_write(out / f"review/{family}_common_2s_3s.csv", [r for r in rows if r[family + "_common_2s_3s"]])
    for sec in (2, 3):
        csv_write(out / f"review/common_three_models_{sec}s.csv", [r for r in rows if r[f"common_three_models_{sec}s"]])
    return rows


def aggregate(out, p):
    result, window_rows, subject_rows, class_rows = {}, [], [], []
    for c in CONDITIONS:
        sec = c.rsplit("_", 1)[1]
        d = np.load(out / f"data_{sec}.npz")
        y, groups, ids = (d[k] for k in ("y", "groups", "record_ids"))
        preds = {arm: np.full(len(y), -1, dtype=int) for arm in p["arms"]}
        fold_metrics, timing = [], {arm: [] for arm in p["arms"]}
        for s in p["outer"]:
            scope = s["id"]
            ex = read(out / scope / "exclusions.json")
            if digest(out / scope / "exclusions.json") != read(out / scope / "exclusions_lock.json")["sha256"]:
                raise ValueError("Exclusion lock changed")
            if set(ex["consensus"]) != set(common_records(ex["record_error_tables"])):
                raise ValueError("Consensus calculation mismatch")
            if any(r.split("/")[0] not in s["train"] for r in ex["consensus"]):
                # Unicode-normalized metadata is the authority; directory names may differ.
                if any(not np.isin(groups[ids == r], s["train"]).all() for r in ex["consensus"]):
                    raise ValueError("Excluded records contain test people")
            row = {"scope": scope}
            for arm in p["arms"]:
                path = out / scope / "evaluate" / c / arm
                m = read(path / "metrics.json")
                if digest(path / "model.joblib") != m["model_sha256"] or digest(path / "predictions.npz") != m["predictions_sha256"]:
                    raise ValueError("Saved model/prediction changed")
                z = np.load(path / "predictions.npz")
                expected = np.flatnonzero(np.isin(groups, s["test"]))
                np.testing.assert_array_equal(z["indices"], expected)
                np.testing.assert_array_equal(z["truth"], y[expected])
                if (preds[arm][expected] >= 0).any():
                    raise ValueError("Duplicate test predictions")
                preds[arm][expected] = z["prediction"]
                if scores(y[expected], z["prediction"]) != m["evaluation"]:
                    raise ValueError("Metric mismatch")
                row[arm] = m
                timing[arm].append(m["predict_ms_per_window_batched"])
            fold_metrics.append(row)
        for arm, pred in preds.items():
            if (pred < 0).any():
                raise ValueError("Incomplete outer OOF")
        metrics = {arm: scores(y, pred) for arm, pred in preds.items()}
        delta = paired_bootstrap(y, preds["before"], preds["filtered"], groups)
        control = paired_bootstrap(y, preds["random_control"], preds["filtered"], groups)
        drops = {label: metrics["filtered"]["per_class_f1"][label] - metrics["before"]["per_class_f1"][label]
                 for label in CLASS_NAMES}
        success = delta["delta"] > 0 and delta["low"] > 0 and min(drops.values()) >= -.05 and control["delta"] > 0
        result[c] = {"metrics": metrics, "filtered_minus_before": delta, "filtered_minus_random": control,
                     "class_f1_delta": drops, "success_criterion_met": success, "fold_metrics": fold_metrics,
                     "mean_batch_predict_ms_per_window": {arm: float(np.mean(t)) for arm, t in timing.items()}}
        np.savez_compressed(out / f"outer_oof_{c}.npz", truth=y, **preds)
        for i in range(len(y)):
            window_rows.append({"condition": c, "record_id": ids[i], "subject": groups[i],
                "start_s": float(d["start_s"][i]), "nominal_end_s": float(d["start_s"][i] + int(sec[0])),
                "truth": CLASS_NAMES[y[i]], **{arm: CLASS_NAMES[pred[i]] for arm, pred in preds.items()}})
        for subject in np.unique(groups):
            mask = groups == subject
            subject_rows.append({"condition": c, "subject": subject, "windows": int(mask.sum()),
                **{arm + "_macro_f1": scores(y[mask], pred[mask])["macro_f1"] for arm, pred in preds.items()}})
        for label in CLASS_NAMES:
            class_rows.append({"condition": c, "label": label,
                **{arm + "_f1": metrics[arm]["per_class_f1"][label] for arm in preds},
                "delta": drops[label], "support": metrics["before"]["per_class"][label]["support"]})
    csv_write(out / "review/window_predictions.csv", window_rows)
    csv_write(out / "review/subject_metrics.csv", subject_rows)
    csv_write(out / "review/class_metrics.csv", class_rows)
    return result


def plots(out, results, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font = Path("/System/Library/Fonts/Supplemental/AppleGothic.ttf")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = "AppleGothic"
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})
    (out / "figures").mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.5), layout="constrained")
    x = np.arange(6)
    for i, (arm, label, color) in enumerate([("before", "제외 전", "#2765a6"), ("filtered", "공통 오분류 제외", "#c37917"), ("random_control", "무작위 제외 대조군", "#aeb6c0")]):
        vals = [results[c]["metrics"][arm]["macro_f1"] * 100 for c in CONDITIONS]
        bars = ax.bar(x + (i - 1) * .25, vals, width=.24, color=color, label=label)
        ax.bar_label(bars, fmt="%.1f", fontsize=9, padding=3)
    ax.set(xticks=x, xticklabels=[title(c).replace(" · ", "\n") for c in CONDITIONS], ylim=(0, 100), ylabel="Macro F1 (%)",
           title="평가 데이터를 그대로 두고 비교한 성능 · 32명, 사람 단위 5분할")
    ax.legend(loc="upper left", ncol=3, frameon=False)
    ax.grid(axis="y", alpha=.15)
    ax.set_axisbelow(True)
    fig.savefig(out / "figures/performance.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    annotation_x = max(results[c]["filtered_minus_before"]["high"] * 100 for c in CONDITIONS) + .25
    for i, c in enumerate(CONDITIONS):
        d = results[c]["filtered_minus_before"]
        ax.errorbar(d["delta"] * 100, i, xerr=[[100 * (d["delta"] - d["low"])], [100 * (d["high"] - d["delta"])]],
                    fmt="o", color="#2765a6", capsize=5)
        ax.text(annotation_x, i, f"{d['delta']*100:+.2f} %p", va="center", fontsize=10)
    ax.axvline(0, color="#444", lw=1)
    ax.set(yticks=np.arange(6), yticklabels=[title(c) for c in CONDITIONS], xlabel="제외 후 - 제외 전 Macro F1 (%p)",
           title="개선 폭의 불확실성 · 사람 단위 재표집 95% 구간")
    ax.invert_yaxis()
    ax.set_xlim(min(results[c]["filtered_minus_before"]["low"] * 100 for c in CONDITIONS) - .4, annotation_x + 1.1)
    fig.savefig(out / "figures/delta.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 5.5), layout="constrained")
    a = np.array([[results[c]["class_f1_delta"][label] * 100 for label in CLASS_NAMES] for c in CONDITIONS])
    lim = max(abs(a).max(), 1)
    im = ax.imshow(a, cmap="PuOr_r", vmin=-lim, vmax=lim, aspect="auto")
    for i in range(6):
        for j in range(7):
            ax.text(j, i, f"{a[i,j]:+.1f}", ha="center", va="center", color="white" if abs(a[i,j]) > lim * .6 else "#222")
    ax.set(xticks=np.arange(7), xticklabels=[x.replace(" ", "\n", 1) for x in CLASS_NAMES], yticks=np.arange(6),
           yticklabels=[title(c) for c in CONDITIONS], title="자세별 F1 변화 · 음수는 해당 자세의 성능 하락")
    fig.colorbar(im, ax=ax, label="F1 변화 (%p)")
    fig.savefig(out / "figures/class_delta.png", dpi=180)
    plt.close(fig)
    # All confusion matrices retain every class and untouched evaluation window.
    for c in CONDITIONS:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.3), layout="constrained")
        vmax = max(np.max(results[c]["metrics"][arm]["confusion_matrix"]) for arm in ("before", "filtered"))
        for ax, arm, name in zip(axes, ("before", "filtered"), ("제외 전", "제외 후")):
            cm = np.array(results[c]["metrics"][arm]["confusion_matrix"])
            ax.imshow(cm, cmap="Blues", vmin=0, vmax=vmax)
            for i in range(7):
                for j in range(7):
                    ax.text(j, i, str(cm[i,j]), ha="center", va="center", fontsize=9, color="white" if cm[i,j] > vmax*.5 else "black")
            ax.set(xticks=range(7), yticks=range(7), xticklabels=[str(i+1) for i in range(7)],
                   yticklabels=list(CLASS_NAMES), xlabel="예측 자세 번호", ylabel="실제 라벨", title=title(c)+" · "+name)
        fig.savefig(out / f"figures/confusion_{c}.png", dpi=160)
        plt.close(fig)


def markdown_to_html(md):
    # Small local renderer for the fixed report grammar; no remote scripts.
    def inline(s):
        s = html.escape(s)
        s = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r'<img alt="\1" src="\2" loading="lazy">', s)
        s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', s)
        s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
        return s
    out, in_table, in_code = [], False, False
    for line in md.splitlines():
        if line.startswith("```"):
            out.append("</code></pre>" if in_code else "<pre><code>")
            in_code = not in_code
            continue
        if in_code:
            out.append(html.escape(line)+"\n")
            continue
        if in_table and not line.startswith("|"):
            out.append("</tbody></table></div>")
            in_table = False
        if line.startswith("|"):
            cells = [x.strip() for x in line.strip("|").split("|")]
            if all(re.fullmatch(r"[-: ]+", x) for x in cells):
                continue
            if not in_table:
                out.append('<div class="table-wrap"><table><thead><tr>'+"".join("<th>"+inline(x)+"</th>" for x in cells)+"</tr></thead><tbody>")
                in_table = True
            else:
                out.append("<tr>"+"".join("<td>"+inline(x)+"</td>" for x in cells)+"</tr>")
        elif line.startswith("#"):
            n = len(line)-len(line.lstrip("#"))
            out.append(f"<h{n}>"+inline(line[n:].strip())+f"</h{n}>")
        elif line.startswith("- "):
            out.append('<p class="bullet">• '+inline(line[2:])+"</p>")
        elif line.strip():
            out.append("<p>"+inline(line)+"</p>")
    if in_table:
        out.append("</tbody></table></div>")
    css = """body{margin:0;background:#f4f6f8;color:#202c38;font:16px/1.8 -apple-system,BlinkMacSystemFont,'Apple SD Gothic Neo',sans-serif}main{max-width:1080px;margin:40px auto;background:white;padding:44px 56px}h1{font-size:32px;line-height:1.35;margin-top:0;letter-spacing:-.7px}h2{font-size:23px;margin:48px 0 18px;padding-top:15px;border-top:1px solid #dde3e9}h3{font-size:18px;margin-top:30px}p{margin:12px 0}a{color:#165f9e;text-decoration-thickness:1px;text-underline-offset:3px}img{display:block;width:100%;height:auto;margin:24px 0}.table-wrap{overflow-x:auto;margin:18px 0}table{border-collapse:collapse;width:100%;font-size:14px}th{background:#edf2f6;text-align:left;white-space:nowrap}th,td{padding:10px 12px;border-bottom:1px solid #dfe5eb}td{vertical-align:top}pre{background:#edf2f6;overflow:auto;padding:18px;font-size:13px}.bullet{padding-left:18px}strong{font-weight:700}@media(max-width:700px){main{margin:0;padding:24px 18px}h1{font-size:26px}h2{font-size:21px}body{font-size:15px}}@media print{body{background:white}main{margin:0;padding:0;max-width:none}h2{break-after:avoid}table,img{break-inside:avoid}a{color:inherit}}"""
    return '<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>공통 오분류 제외 재학습 실험</title><style>'+css+'</style><main>'+"\n".join(out)+"</main></html>"


def review_page(out, rows):
    import json
    dest = ROOT / "docs/research/results" / out.name / "review.html"
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = [{**r, "source_relative": str(Path(r["source_file"]).relative_to(ROOT))} for r in rows]
    document = '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>공통 오분류 기록 검토</title><script>if(location.pathname.includes('/research/results/')&&!location.pathname.includes('/docs/research/results/')){const b=document.createElement('base');b.href=location.href.replace('/research/results/','/docs/research/results/');document.head.append(b)}</script><style>
body{margin:0;background:#f4f6f8;color:#202c38;font:16px/1.6 -apple-system,BlinkMacSystemFont,'Apple SD Gothic Neo',sans-serif}main{max-width:1050px;margin:auto;padding:30px 22px}h1{font-size:29px;margin:12px 0}h2{font-size:18px;margin:0}p{margin:10px 0}.muted{color:#5b6874;font-size:14px}.controls{display:flex;gap:12px;flex-wrap:wrap;margin:24px 0}label{font-size:13px;display:grid;gap:5px;flex:1;min-width:180px}select,input,button{font:inherit;padding:10px;border:1px solid #b8c3ce;border-radius:5px;background:white;color:inherit}button{cursor:pointer}button:disabled{opacity:.45;cursor:default}article{background:white;padding:20px;margin:14px 0;border:1px solid #dbe2e8;border-radius:8px}summary{cursor:pointer;color:#165f9e;margin:12px 0}.table{overflow:auto}table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;padding:8px;border-bottom:1px solid #e2e7ec}a{color:#165f9e;overflow-wrap:anywhere}nav{display:flex;align-items:center;gap:15px;margin:24px 0}.badge{color:#84500c;background:#fff2db;display:inline-block;padding:2px 8px;font-size:13px}#count{font-weight:600}@media(max-width:600px){main{padding:18px 12px}article{padding:15px}h1{font-size:24px}}</style>
<main><a href="../../../reports/RUN_report.html">← 실험 보고서</a><h1>공통 오분류 기록 검토</h1>
<p>사람 단위 교차검증에서 반복해서 틀린 기록입니다. 실제 라벨 오류나 수집 이상으로 확정된 목록은 아닙니다.</p>
<div class="controls"><label>공통 범위<select id="scope"><option value="common_all_six">여섯 조건 모두</option><option value="random_forest_common_2s_3s">Random Forest · 2초·3초</option><option value="rbf_svm_common_2s_3s">RBF SVM · 2초·3초</option><option value="xgboost_common_2s_3s">XGBoost · 2초·3초</option><option value="common_three_models_2s">2초 · 세 모델</option><option value="common_three_models_3s">3초 · 세 모델</option><option value="all">전체 기록</option></select></label><label>자세<select id="label"><option value="">전체 자세</option></select></label><label>사람·파일 검색<input id="search" type="search" placeholder="이름 또는 파일명"></label></div>
<p id="count" aria-live="polite"></p><p class="muted">공통 기준: 각 조건에서 해당 기록 구간의 50% 이상을 오분류. 평균 오류율 내림차순이며, 실제 수집 오류 확률의 순위가 아닙니다.</p>
<div id="records"></div><nav><button id="prev">이전</button><span id="page"></span><button id="next">다음</button></nav>
<p><a href="../../../../research/results/RUN/review/common_all_six.csv">여섯 조건 공통 CSV</a> · <a href="../../../../research/results/RUN/review/all_record_errors.csv">전체 오류표 CSV</a></p></main>
<script>const DATA=PAYLOAD;const CONDITIONS=CONDITIONS_JSON;const LABELS=LABELS_JSON;let page=0;const size=20;const $=id=>document.getElementById(id);const el=(tag,text)=>{const e=document.createElement(tag);if(text!==undefined)e.textContent=text;return e;};for(const label of LABELS){const o=el('option',label);o.value=label;$('label').append(o)}
function render(){const key=$('scope').value,label=$('label').value,q=$('search').value.trim().normalize('NFC').toLowerCase();const data=DATA.filter(r=>(key==='all'||r[key])&&(!label||r.truth_label===label)&&(!q||(r.subject+' '+r.record_id).normalize('NFC').toLowerCase().includes(q)));data.sort((a,b)=>b.mean_error_rate-a.mean_error_rate||a.record_id.localeCompare(b.record_id));page=Math.max(0,Math.min(page,Math.ceil(data.length/size)-1));$('count').textContent=data.length+'개 기록 · '+new Set(data.map(r=>r.subject)).size+'명';$('records').replaceChildren();for(const r of data.slice(page*size,(page+1)*size)){const card=el('article');card.append(el('h2',r.subject+' · '+r.truth_label));const file=el('a',r.record_id);file.href='../../../../'+r.source_relative.split('/').map(encodeURIComponent).join('/');card.append(file);const note=el('p','평균 오분류 '+(100*r.mean_error_rate).toFixed(1)+'% · '+r.duration_s.toFixed(1)+'초 · 내부 탐색 제외 '+r.nested_exclusion_count+'/'+r.nested_training_opportunities+'회');card.append(note);const badge=el('span',r.pattern+' · 미확인');badge.className='badge';card.append(badge);const details=el('details');details.append(el('summary','모델별 오류율 보기'));const wrap=el('div');wrap.className='table';const table=el('table');const header=el('tr');for(const t of ['모델·시간','오분류율','구간 수','대표 예측'])header.append(el('th',t));table.append(header);for(const [key,name] of CONDITIONS){const tr=el('tr');for(const t of [name,(r[key+'_error_rate']*100).toFixed(1)+'%',String(r[key+'_windows']),r[key+'_dominant_prediction']])tr.append(el('td',t));table.append(tr)}wrap.append(table);details.append(wrap,el('p','SHA-256: '+r.source_sha256));card.append(details);$('records').append(card)}$('page').textContent=(data.length?page+1:0)+' / '+Math.ceil(data.length/size);$('prev').disabled=page===0;$('next').disabled=(page+1)*size>=data.length;}
for(const id of ['scope','label','search'])$(id).addEventListener('input',()=>{page=0;render()});$('prev').onclick=()=>{page--;render();window.scrollTo(0,0)};$('next').onclick=()=>{page++;render();window.scrollTo(0,0)};render();</script></html>'''
    document = document.replace("RUN", out.name).replace("PAYLOAD", json.dumps(payload, ensure_ascii=False).replace("</", "<\\/"))
    document = document.replace("CONDITIONS_JSON", json.dumps([[c, title(c)] for c in CONDITIONS], ensure_ascii=False)).replace("LABELS_JSON", json.dumps(list(CLASS_NAMES), ensure_ascii=False))
    dest.write_text(document, encoding="utf-8")
    alias = out / "review.html"
    if not alias.exists():
        import os
        alias.symlink_to(os.path.relpath(dest, alias.parent))
    return dest


def report(out, p, results, rows):
    common = [r for r in rows if r["common_all_six"]]
    passed = sum(r["success_criterion_met"] for r in results.values())
    rel = "../../research/results/" + out.name
    image = lambda name: f"![{name}]({rel}/figures/{name}.png)"
    deltas = [r["filtered_minus_before"]["delta"] * 100 for r in results.values()]
    improved = sum(d > 0 for d in deltas)
    directional = ("여섯 조건 모두에서 공통 후보 제외 후 점수가 낮아졌다." if improved == 0 else
                   "여섯 조건 모두에서 공통 후보 제외 후 점수가 높아졌다." if improved == 6 else
                   f"점수가 높아진 조건은 {improved}개, 낮아지거나 같은 조건은 {6-improved}개로 효과가 일관되지 않았다.")
    lines = ["# 공통 오분류를 제외하면 자세 분류가 더 좋아질까?", "",
        "## 핵심 결과", "",
        f"Random Forest·RBF SVM·XGBoost를 각각 2초·3초로 학습하고, 여섯 조건에서 공통으로 어려운 기록을 제외해 다시 학습했다. **원본 기록은 삭제하지 않았다.** 사용자가 적은 ‘RBM SVM’은 기존 프로젝트의 RBF 커널 SVM을 의미하는 것으로 해석했다.", "",
        f"- 실험 시작 시 고정한 적격 자료는 **{len(p['subjects'])}명·{p['records']}개 기록·7개 자세**다. 2초는 {p['cohorts']['2']['windows']:,}개, 3초는 {p['cohorts']['3']['windows']:,}개 구간이다.",
        f"- 전체 자료의 사람 단위 교차검증에서 **{len(common)}개 기록({len(common)/len(rows)*100:.1f}%)**이 여섯 조건 모두의 공통 오분류 후보였다. ‘잘못 수집된 데이터’로 확인된 수가 아니다.",
        f"- 학습 집합 안에서만 후보를 찾아 제외한 공정한 비교에서, Macro F1 변화는 **{min(deltas):+.2f}~{max(deltas):+.2f}%p**였다. 사전에 정한 개선 기준을 충족한 조건은 **6개 중 {passed}개**다.",
        f"- {directional}",
        "- 삭제의 유용성은 새로운 사람에게도 맞히는지를 기준으로 판단해야 한다. 후보 목록은 라벨·자세·센서 설치 상태를 확인할 검토 목록이며, 자동 삭제 목록으로 사용하지 않는 것이 적절하다.", "",
        f"[사람·자세·모델별로 후보 기록 찾아보기](../research/results/{out.name}/review.html)", "",
        "**자료 기준 시점:** 실험 도중 별도 작업에서 7개 기록의 학습 제외 설정과 관련 코드가 추가되어, 보호 장치가 남은 학습을 중단했다. 이후 저장해 둔 시작 시점의 소스·특징으로 이어서 완료했다. 아래 수치는 고정한 735개 기록 기준이며, 새 7건 제외 정책을 적용한 현재 학습 집합의 결과는 아니다. 해당 별도 변경은 되돌리거나 덮어쓰지 않았다.", "",
        "## 1. 어떤 실험을 했는가", "",
        "Macro F1은 일곱 자세의 F1 점수를 같은 비중으로 평균한 값이다. 한 자세만 잘 맞혀 전체 정답률이 높아지는 것을 줄여 보려는 지표다. 여기서 ‘+1%p’는 예를 들어 65%가 66%가 되는 변화를 뜻한다.", "",
        "| 단계 | 수행 내용 | 왜 필요한가 |", "| --- | --- | --- |",
        "| ① 외부 평가 분할 | 32명을 5묶음으로 나누고, 매번 한 묶음을 평가에만 사용 | 같은 사람의 유사한 구간이 학습과 평가에 섞이는 누수 방지 |",
        "| ② 학습 내부 탐색 | 남은 학습 사람들만 다시 3분할하여 각 기록의 미학습 예측 생성 | 자기 학습 데이터를 외운 결과로 이상치를 정하지 않음 |",
        "| ③ 제외 기준 고정 | 한 기록의 구간 중 50% 이상 오분류가 여섯 모델·시간 조건 모두에서 발생 | 시간 길이가 달라도 동일 원본 파일로 공통 여부 비교 |",
        "| ④ 재학습 비교 | 제외 전 / 공통 후보 제외 / 자세별 같은 개수 무작위 제외 | 데이터 축소 자체의 영향과 대상 선택의 효과 비교 |",
        "| ⑤ 동일 평가 | 평가 사람·기록·어려운 구간을 전혀 빼지 않고 세 방법 평가 | 쉬운 평가 자료만 남겨 성능을 부풀리는 선택 편향 방지 |",
        "| ⑥ 전체 자료 재학습 | 교차검증을 끝낸 후 전체 자료의 후보 목록으로 6개 전·6개 후 모델 저장 | 검토·후속 실험용이며 독립 평가를 새로 얻는 것은 아님 |", "",
        "2초와 3초는 모두 12 Hz로 보간하고, 비중첩 구간에서 596개 특징을 추출했다. 2초는 24개 시점, 3초는 36개 시점이다. 마지막 구간은 명목 길이의 95% 이상일 때만 유지한다. 같은 원본 735개를 쓰지만 구간 수와 말미 포함 범위가 달라, 시간 길이의 차이만 완전히 분리한 실험은 아니다.", "",
        "기존 학습 적격 정책과 라벨 교정 4건을 따랐다. 명시적인 측정 전용·학습 제외·등록/보정 자료는 사용하지 않는다. 예전 개인화 실험에서 임시로 예약했던 첫 정자세 파일을 이번 일반 분류에서 인위적으로 빼지는 않았다. 원본과 기존 모델은 SHA-256·크기·수정시각으로 보존 여부를 확인했다.", "",
        "### 고정한 모델 설정", "",
        "| 모델 | 설정 |", "| --- | --- |",
        "| Random Forest | 나무 200개, min_samples_leaf=1, max_features=sqrt, class_weight=balanced |",
        "| RBF SVM | C=1, gamma=0.0005, class_weight=balanced, probability=True |",
        "| XGBoost | 나무 200개, depth=3, learning_rate=0.05, min_child_weight=3, L2=5, 행·열 샘플링 각 0.8, 학습 집합 클래스 균형 가중치 |", "",
        "모든 모델은 각 학습 부분집합에서만 StandardScaler를 학습하며 predict_proba의 최댓값으로 분류했다. SVM의 내부 확률 보정은 현재 학습 행 안에서만 이루어지지만 사람 단위 확률 보정은 아니다. 이번에는 하이퍼파라미터·임계값 탐색을 하지 않았다. 결과는 이 설정과 한 번의 5분할에 조건부인 비교다.", "",
        "## 2. 제외 전후 성능", "",
        "아래 점수는 각 사람이 자신을 학습하지 않은 모델로 평가된 예측을 합친 값이다. 매 분할의 학습 인원은 25~26명, 평가 인원은 6~7명이다. 전체 32명을 학습한 최종 모델의 새로운 사람에 대한 점수와는 다르다.", "",
        "| 모델·구간 | 제외 전 F1 | 공통 제외 F1 | 변화 (%p) | 변화의 95% 구간 | 무작위 제외 F1 | 기준 충족 |",
        "| --- | ---: | ---: | ---: | --- | ---: | --- |"]
    for c, r in results.items():
        m, d = r["metrics"], r["filtered_minus_before"]
        lines.append(f"| {title(c)} | {m['before']['macro_f1']*100:.2f}% | {m['filtered']['macro_f1']*100:.2f}% | {d['delta']*100:+.2f} | [{d['low']*100:+.2f}, {d['high']*100:+.2f}] | {m['random_control']['macro_f1']*100:.2f}% | {'충족' if r['success_criterion_met'] else '미충족'} |")
    lines += ["", image("performance"), "", image("delta"), "",
        "95% 구간은 사람 전체를 묶어 2,000회 짝지어 재표집한 값이다. 0을 포함하면 이 평가에서 개선·악화를 명확히 구분하기 어렵다. 같은 데이터로 학습한 분할 간 상관과 모델을 다시 학습할 때의 변동을 모두 반영하지는 않으며, 여섯 비교의 다중검정 보정도 하지 않았다. 따라서 확정적인 통계적 입증으로 해석하지 않는다.", "",
        "사전 개선 기준은 ① F1 증가 ② 변화 구간 하한 > 0 ③ 어떤 자세도 F1이 5%p 초과 하락하지 않음 ④ 무작위 제외보다 F1 높음이다. 무작위 대조군은 자세별 기록 수를 맞추지만 사람별 수·구간 수·시간 길이까지 맞추지는 않으며, 한 번의 무작위 선택이므로 제한적 대조다.", "",
        "| 모델·구간 | 제외 전 Accuracy | 제외 후 Accuracy | 제외 전 최저 자세 F1 | 제외 후 최저 자세 F1 | 공통 제외 − 무작위 (%p) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for c, r in results.items():
        a, b = r["metrics"]["before"], r["metrics"]["filtered"]
        lines.append(f"| {title(c)} | {a['accuracy']*100:.2f}% | {b['accuracy']*100:.2f}% | {a['worst_class_f1']*100:.2f}% | {b['worst_class_f1']*100:.2f}% | {r['filtered_minus_random']['delta']*100:+.2f} |")
    worst = min(((delta, c, label) for c, r in results.items() for label, delta in r["class_f1_delta"].items()))
    lines += ["", f"가장 큰 자세별 하락은 **{title(worst[1])}의 {worst[2]}에서 {worst[0]*100:+.2f}%p**였다. 전체 평균이 비슷해도 이런 손실을 함께 확인해야 한다."]
    lines += ["", "### 자세별로 좋아진 곳과 나빠진 곳", "", image("class_delta"), "",
              f"[자세별 F1 전체 표]({rel}/review/class_metrics.csv) · [사람별 점수]({rel}/review/subject_metrics.csv)", "",
              "## 3. 어떤 데이터를 공통으로 틀렸는가", "",
              "다음 목록은 전체 자료에 대한 제외 전 사람 단위 교차검증으로 만들었다. 각 평가 분할의 실제 학습 제외 목록은 학습 내부 3분할에서 따로 만들었으므로, 아래 전체 목록을 교차검증에 소급 적용하지 않았다.", "",
              "| 공통 범위 | 후보 기록 수 | 검토 목록 |", "| --- | ---: | --- |"]
    for family in MODELS:
        n = sum(r[family + "_common_2s_3s"] for r in rows)
        lines.append(f"| {NAMES[family]}의 2초·3초 모두 | {n} | [CSV]({rel}/review/{family}_common_2s_3s.csv) |")
    for sec in (2, 3):
        n = sum(r[f"common_three_models_{sec}s"] for r in rows)
        lines.append(f"| {sec}초의 세 알고리즘 모두 | {n} | [CSV]({rel}/review/common_three_models_{sec}s.csv) |")
    lines += [f"| 여섯 조건 모두 · 재학습 제외 규칙 | {len(common)} | [공통 후보 CSV]({rel}/review/common_all_six.csv) |", "",
        "각 CSV에는 원본 파일 경로·SHA-256·사람·자세·여섯 조건의 오분류율·대표 예측 자세가 있다. 대표 예측이 동률이면 클래스 순서상 앞선 자세를 표시하므로, 동률을 확정된 다른 자세로 해석하지 않는다. 구간별 시작·끝 시각과 세 실험군의 예측은 별도 전체 예측 CSV에서 확인한다.", "",
        "### 자세별 후보 비율", "", "| 실제 라벨 | 전체 기록 | 공통 후보 | 후보 비율 |", "| --- | ---: | ---: | ---: |"]
    for label in CLASS_NAMES:
        total = sum(r["truth_label"] == label for r in rows)
        n = sum(r["truth_label"] == label for r in common)
        lines.append(f"| {label} | {total} | {n} | {n/total*100:.1f}% |")
    people = sorted({r["subject"] for r in common}, key=lambda s: (-sum(r["subject"] == s for r in common), s))
    lines += ["", "### 후보가 많은 사람 상위 10명", "", "| 사람 | 전체 기록 | 후보 기록 | 후보 비율 |", "| --- | ---: | ---: | ---: |"]
    for subject in people[:10]:
        n = sum(r["subject"] == subject for r in common)
        total = sum(r["subject"] == subject for r in rows)
        lines.append(f"| {subject} | {total} | {n} | {n/total*100:.1f}% |")
    same = sum(r["pattern"] == "같은 다른 자세로 일치" for r in common)
    lines += ["", f"공통 후보 중 {same}개는 여섯 조건의 대표 예측이 같은 다른 자세였고, {len(common)-same}개는 예측이 갈리거나 동률 문제가 있었다. 첫 경우도 라벨 오류의 증거 자체는 아니다. 체형·앉는 방식·센서 배치가 다른 사람에게 여러 알고리즘이 같은 편향을 가질 수 있다.", "",
        "### 우선 확인할 기록 12개", "", "평균 오분류율 내림차순이다. 실제 오류 확률이나 수집 품질의 순위는 아니다.", "",
        "| 원본 파일 | 실제 라벨 | 여섯 조건 평균 오분류율 | 패턴 |", "| --- | --- | ---: | --- |"]
    for r in common[:12]:
        source_path = Path(r["source_file"])
        display = str(source_path.relative_to(ROOT / "data"))
        from urllib.parse import quote
        lines.append(f"| [{display}](../../data/{quote(display)}) | {r['truth_label']} | {r['mean_error_rate']*100:.1f}% | {r['pattern']} |")
    lines += ["", f"[전체 735개 기록의 오류표]({rel}/review/all_record_errors.csv) · [모든 평가 구간 예측·시간 위치]({rel}/review/window_predictions.csv)", "",
        "### 각 평가 분할에서 실제 제외한 양", "",
        "| 분할 | 학습 인원 | 평가 인원 | 공통 제외 기록 | 학습 기록 | 2초 제외 구간 | 3초 제외 구간 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for s in p["outer"]:
        ex = read(out / s["id"] / "exclusions.json")
        a = results["random_forest_2s"]["fold_metrics"][int(s["id"][-1])]
        b = results["random_forest_3s"]["fold_metrics"][int(s["id"][-1])]
        lines.append(f"| {s['id']} | {len(s['train'])} | {len(s['test'])} | {len(ex['consensus'])} | {a['before']['train_records']} | {a['before']['train_windows']-a['filtered']['train_windows']} | {b['before']['train_windows']-b['filtered']['train_windows']} |")
    lines += ["", "각 분할의 후보 수가 전체 목록과 다른 것은 정상이다. 내부 탐색 모델의 학습 인원과 구성도 달라지므로, 후보 판정 자체가 데이터에 민감하다는 점까지 포함해 평가한 것이다.", "",
        f"전체 공통 후보 {len(common)}개 중 학습 집합에 속한 네 번 모두 내부 탐색에서 제외된 기록은 {sum(r['nested_exclusion_count'] == r['nested_training_opportunities'] for r in common)}개다. CSV의 nested_exclusion_count에서 기록별 반복성을 확인할 수 있다. 4/4라고 해도 실제 수집 오류라는 의미는 아니다.", "",
        "## 4. 전체 자료로 다시 학습한 최종 연구 모델", "",
        f"제외 전 {p['records']}개와 공통 후보 {len(common)}개를 제외한 {p['records']-len(common)}개 기록으로 여섯 조건을 각각 학습했다. 총 12개 최종 연구 모델을 저장했다. 이 모델들은 승인·활성화하지 않았다. 아래 ‘학습 F1’은 해당 모델에 실제 사용한 기록으로 재평가한 값이며, 제외 전후의 평가 집합 자체도 달라지므로 일반화 개선의 근거가 아니다.", "",
        "| 모델·구간 | 제외 전 학습 F1 | 제외 후 남은 자료 학습 F1 | 제외 후 모델의 원래 전체 자료 F1 | 모델 파일 |",
        "| --- | ---: | ---: | ---: | --- |"]
    for c in CONDITIONS:
        a = read(out / "full/evaluate" / c / "before/metrics.json")
        b = read(out / "full/evaluate" / c / "filtered/metrics.json")
        lines.append(f"| {title(c)} | {a['training']['macro_f1']*100:.2f}% | {b['training']['macro_f1']*100:.2f}% | {b['evaluation']['macro_f1']*100:.2f}% | [전]({rel}/full/evaluate/{c}/before/model.joblib) · [후]({rel}/full/evaluate/{c}/filtered/model.joblib) |")
    lines += ["", "## 5. 이 방식은 적절한가", "",
        "**문제 후보를 찾는 보조 도구로는 적절하지만, 공통 오분류만으로 원본을 삭제하는 방식은 권장하기 어렵다.** 세 알고리즘이 모두 틀려도 동일한 센서 특징·라벨·학습 집합을 공유하므로 독립적인 세 사람의 판정과 같지 않다. 특히 드문 체형이나 특이한 압력 분포를 가진 정상 기록을 없애면 제품이 지원해야 할 사람 범위를 좁힐 수 있다.", "",
        f"이번 공정 비교에서 사전 기준을 통과한 조건은 {passed}/6개다. 관측된 점수의 방향뿐 아니라 사람별 변동, 특정 자세 손실, 무작위 제외 대조군을 함께 보아야 한다. 기존 모델의 활성 상태를 바꿀 근거로 삼지 않았다.", "",
        "- 실제 오류라면: 수집 당시 영상·관찰 기록·동기화 로그로 확인하고, 라벨 오류는 근거를 남겨 교정한다. 특정 시간만 문제라면 전체 파일을 버리기 전에 구간 단위로 처리한다.",
        "- 실제 어려운 사례라면: 삭제보다 해당 사람·자세·압력 패턴을 더 수집하고, 개인화 또는 불확실 응답을 검토한다.",
        "- 증거가 없다면: 미확인 후보로 유지하고 재수집한다. 모델의 다수결을 정답 라벨로 덮어쓰지 않는다.", "",
        "50%라는 임계값과 여섯 조건 일치 규칙은 실험 전에 정한 탐색 규칙이며, 최적이라는 근거는 없다. 단 한 번의 제외만 수행했고 반복해서 어려운 자료를 지우지 않았다. 전체 파일 제외는 정상 구간도 함께 잃는 거친 조치다.", "",
        "## 6. 앞으로의 개선 방향과 후속 실험", "",
        "| 우선순위 | 수행할 일 | 검증 방법 |", "| --- | --- | --- |",
        "| 1 · 원인 확인 | 공통 후보를 영상·관찰·센서 로그와 대조해 라벨 오류 / 전환 구간 / 설치 변화 / 정상 희귀 사례로 분류 | 모델 예측만으로 확정하지 않고, 판단 근거와 검토자를 별도 기록 |",
        "| 2 · 재측정 | 후보 비율이 높은 사람·자세와 정상 대조군을 같은 의자에서 다시 수집. 자리에서 일어나 재착석하고 설치도 반복 | 사람·세션·의자·설치 식별자, 실제 자세 시작·종료 시각, 비착석 기준 압력 저장 |",
        "| 3 · 처리 방식 비교 | 원본 유지 / 확인된 라벨만 교정 / 확인된 불량 구간만 제외 / 가중치 완화 비교 | 같은 외부 사람 분할과 고정 평가 집합 유지. 각 분할의 내부에서만 처리 기준 학습 |",
        "| 4 · 특징·시간 실험 | 총압력·상대압력·압력중심·개인 기준 보정과 2초·3초를 비교 | 시간 길이 외 요인을 고정하고 사람 단위 반복 분할. SVM 확률 보정도 사람 단위로 비교 |",
        "| 5 · 독립 검증 | 이번 자료와 겹치지 않는 새 사람·새 세션을 미리 보류 | 모델·임계값·제외 규칙을 잠근 뒤 딱 한 번 평가. Macro F1, 자세별·사람별 성능과 신뢰구간 보고 |",
        "| 6 · 실시간 검증 | 실제 센서와 GUI에서 자세 전환·앉기/일어서기·지연을 확인 | 2초/3초 취득 대기 + 특징 계산 + 추론 + 알림 지연을 각각 측정 |", "",
        "현재 배치 예측 시간은 수천 구간을 한 번에 처리한 조건의 값이므로 실시간 1회 추론 지연이나 알림 지연으로 대체할 수 없다. 새 사람 수는 임의의 ‘충분한 인원’ 숫자로 확정하지 말고, 사람별 변동과 목표 신뢰구간 폭을 이용해 사전 표본 계획을 세우는 것이 좋다.", "",
        "## 7. 재현성·검증 범위·근거", "",
        f"- 실제 실행 시작: {p['started_at']}",
        f"- 학습 종료: {read(out / 'execution_complete.json')['finished_at']}",
        f"- 보고서 생성: {now()}",
        f"- 소스 revision: {p['source_revision']}. 시작 시 기존 미커밋 변경이 있었으며 protocol에 원문을 보존했다.",
        "- 센서 측정일은 이번 학습 실행일과 다르다. 파일 수정시각으로 측정일을 추정하지 않았다.",
        "- 총 192회 fit: 내부 후보 탐색 90회, 외부 평가 비교 90회, 전체 자료 전후 모델 12회. 외부·전체 모델은 저장 후 재로딩 예측 일치를 확인했다.",
        "- 원본 data 전체와 시작 시 이미 존재한 research/models 파일의 SHA-256·크기·mtime는 동일했다. research/models에는 다른 작업의 training_exclusions.json 한 파일이 새로 추가되어 전체 폴더가 완전히 동일하지는 않다. 이번 연구 모델은 결과 폴더에만 저장했다.",
        f"- 자동 테스트: 기존 단위 테스트 {read(out / 'verification.json')['tests']['unit_tests.log']['count']}개, 실험 규칙·분석 테스트 {read(out / 'verification.json')['tests']['experiment_tests.log']['count']}개 통과. 외부 평가 예측의 누락·중복, 동일 평가 집합 유지, 저장 모델·예측 해시와 지표 재계산을 검사했다.",
        "- 이번 검증은 과거 CSV 기반 재학습·사람 분리 평가다. 실제 센서, 실제 Tk GUI, macOS 재빌드·서명·Gatekeeper, 새 모집 집단의 독립 검증은 수행하지 않았다.",
        "- 거의 일정한 채널에서 scipy의 왜도·첨도 계산 경고가 발생했다. 기존 특징 추출기는 비유한 값을 0으로 치환하며 최종 특징의 유한성을 검사했다. 이 처리 자체가 센서 고장 여부를 검증한 것은 아니다.", "",
        f"[고정 실험 계획]({rel}/protocol.json) · [평가 결과 JSON]({rel}/summary.json) · [검증 기록]({rel}/verification.json) · [원본·교정 복사본 대응]({rel}/training_data/staged_sources.json)", "",
        f"[동시 작업 변경과 원본 보존 점검]({rel}/concurrent_workspace_changes.json)", "",
        "재실행 명령:", "", "```sh",
        ".venv/bin/python -m experiments.consensus_filter.run --output research/results/<새-실행-폴더>",
        ".venv/bin/python -m experiments.consensus_filter.analyze --output research/results/<새-실행-폴더>", "```", "",
        "새 실행은 그 시점의 데이터와 제외 정책을 적용하므로 이번 735개와 다른 집합이 될 수 있다. 이번 결과를 다시 검산할 때는 analyze의 --output에 이번 기존 실행 폴더를 지정한다. 학습 당시 소스는 source_snapshot, 실제 입력·교정 이력은 training_data와 audit.json, 고정 특징은 data_2s.npz·data_3s.npz에 보존했다. 동시 작업 이후 남은 학습은 resume_frozen.py로 이 소스 스냅샷에서 실행했다.", "",
        "분석 근거: 학습과 평가를 먼저 나누고 전처리·선택을 학습 데이터 안에서 수행해야 한다는 원칙은 [scikit-learn의 데이터 누수 안내](https://scikit-learn.org/stable/common_pitfalls.html)에 따른다. 동일 사람의 반복 기록은 독립 표본으로 무작위 분할하지 않고 [그룹 교차검증](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data)으로 분리한다. 오분류 후보와 통계적 이상치의 구분은 [이상치·새로운 관측 탐지 안내](https://scikit-learn.org/stable/modules/outlier_detection.html)를 참고했다. 이 실험의 수치는 외부 문헌이 아닌 저장된 실제 예측에서 계산했다.", "",
        "## 부록 · 여섯 조건의 혼동행렬", "",
        "가로는 예측, 세로는 실제 라벨이다. 자세 번호는 위에서부터 1~7 순서이며 각 그림의 세로축 자세 순서와 같다. 모든 평가 구간을 포함한다.", ""]
    for c in CONDITIONS:
        lines += [f"### {title(c)}", "", image("confusion_"+c), ""]
    path = ROOT / "docs/reports" / f"{out.name}_report.md"
    md = "\n".join(lines)
    path.write_text(md, encoding="utf-8")
    path.with_suffix(".html").write_text(markdown_to_html(md), encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    p = verify_frozen(out)
    if not (out / "execution_complete.json").is_file():
        raise ValueError("Training incomplete")
    results = aggregate(out, p)
    rows = review_lists(out, p)
    review_page(out, rows)
    plots(out, results, rows)
    protection = read(out / "concurrent_workspace_changes.json")
    unchanged = not protection["protected_inventory_changes"]
    for path in out.glob("full/evaluate/*/*/metrics.json"):
        m = read(path)
        if digest(path.parent / "model.joblib") != m["model_sha256"] or digest(path.parent / "predictions.npz") != m["predictions_sha256"]:
            raise ValueError("Full model artifact changed")
        condition, arm = path.parents[1].name, path.parent.name
        d = np.load(out / ("data_" + condition.rsplit("_", 1)[1] + ".npz"))
        z = np.load(path.parent / "predictions.npz")
        np.testing.assert_array_equal(z["indices"], np.arange(len(d["y"])))
        np.testing.assert_array_equal(z["truth"], d["y"])
        if scores(d["y"], z["prediction"]) != m["evaluation"]:
            raise ValueError("Full original-cohort metric mismatch")
        kept = np.ones(len(d["y"]), dtype=bool) if arm == "before" else ~np.isin(d["record_ids"], read(out / "full/exclusions.json")["consensus"])
        if scores(d["y"][kept], z["prediction"][kept]) != m["training"]:
            raise ValueError("Full retained-cohort training metric mismatch")
    inner_fit_count = 0
    for path in out.glob("outer_*/inner/*/metrics.json"):
        m = read(path)
        if digest(path.parent / "predictions.npz") != m["predictions_sha256"]:
            raise ValueError("Inner discovery predictions changed")
        inner_fit_count += len(m["fit_seconds"])
    saved_count = len(list(out.glob("**/model.joblib")))
    if inner_fit_count != 90 or saved_count != 102:
        raise ValueError("Unexpected completed fit counts")
    verification = {"checked_at": now(), "protected_inputs_models_unchanged": unchanged,
        "canonical_data_unchanged": True, "preexisting_model_artifacts_unchanged": True,
        "concurrent_added_exclusion_policy_preserved": bool(protection["protected_inventory_changes"]),
        "outer_evaluation_full_coverage": True, "outer_test_unchanged_across_arms": True,
        "all_saved_models_and_prediction_hashes_verified": True,
        "outer_metrics_recomputed": True, "exclusions_training_only": True,
        "fit_count": inner_fit_count + saved_count, "inner_discovery_fits": inner_fit_count, "saved_models": saved_count,
        "analysis_source_sha256": digest(Path(__file__)), "unit_tests_log": "unit_tests.log"}
    verification["tests"] = {}
    for name in ("unit_tests.log", "experiment_tests.log"):
        log = (out / name).read_text()
        match = re.search(r"Ran (\d+) tests", log)
        if not match or not log.rstrip().endswith("OK"):
            raise ValueError(f"Tests not successful: {name}")
        verification["tests"][name] = {"count": int(match[1]), "status": "passed", "sha256": digest(out / name)}
    analysis_source = out / "analysis_source"
    analysis_source.mkdir(exist_ok=True)
    shutil.copy2(Path(__file__), analysis_source / "analyze.py")
    write_json(out / "verification.json", verification)
    write_json(out / "summary.json", {"generated_at": now(), "results": results,
        "common_all_six_records": sum(r["common_all_six"] for r in rows), "protocol_sha256": digest(out / "protocol.json")})
    path = report(out, p, results, rows)
    print("REPORT", path)
    for c, r in results.items():
        print(c, {arm: round(v["macro_f1"], 5) for arm, v in r["metrics"].items()}, r["filtered_minus_before"])


if __name__ == "__main__":
    main()
