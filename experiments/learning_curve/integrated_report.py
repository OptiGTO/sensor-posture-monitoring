"""One six-section report: only out-of-person predictions measure performance."""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
from experiments.learning_curve.integrated import assert_no_leakage
from experiments.learning_curve.resume_frozen import verify_evidence as verify, verify_originals
from experiments.learning_curve.run import ROOT, MODELS, SIZES, SEED, read, digest, now, protected_snapshot
from experiments.learning_curve.report import cm_metrics, forecast, paired_gains, plot, NAMES, COLORS, fit_curve
from src.model_comparison import write_json
from src.posture_core import CLASS_NAMES


def collect(out):
    p = verify(out)
    if not (out / "training_complete.json").exists():
        raise ValueError("Wait for all held-out fits")
    d = np.load(out / "data.npz", allow_pickle=False)
    subjects = np.unique(d["groups"]); sid = np.searchsorted(subjects, d["groups"])
    draws = np.random.default_rng(SEED).integers(len(subjects), size=(2000, len(subjects)))
    summary, bootstrap, csv_rows = {}, {}, []
    arms = [a["id"] for a in p["splits"][0]["arms"]] + ["subjects_31"]
    cubes = {}
    for family in MODELS:
        summary[family], bootstrap[family] = {}, {}
        for arm in arms:
            selected = [s for s in p["splits"] if any(a["id"] == arm for a in s["arms"])]
            repeats = 1 if arm == "subjects_31" else 2
            cm = np.zeros((repeats, len(subjects), 7, 7), dtype=int)
            seen = np.zeros((repeats, len(d["y"])), dtype=int)
            fits = []
            for split in selected:
                dest = out / "fits" / family / split["id"] / arm
                m = read(dest / "metrics.json")
                if m["score_kind"] != "heldout_subjects":
                    raise ValueError("Resubstitution metrics cannot enter performance curves")
                for name, key in [("model.joblib", "model_sha256"), ("predictions.npz", "predictions_sha256")]:
                    if digest(dest / name) != m[key]:
                        raise ValueError("Artifact changed")
                pred = np.load(dest / "predictions.npz", allow_pickle=False)
                ix = pred["indices"]
                np.testing.assert_array_equal(ix, split["evaluation_indices"])
                np.testing.assert_array_equal(pred["truth"], d["y"][ix])
                arm_spec = next(a for a in split["arms"] if a["id"] == arm)
                assert_no_leakage(d, np.array(arm_spec["train_indices"]), ix)
                r = split["repeat"]
                np.add.at(cm[r], (sid[ix], pred["truth"], pred["prediction"]), 1)
                seen[r, ix] += 1; fits.append(m)
                csv_rows.append({k: m[k] for k in ["family", "split", "arm", "subjects", "records", "windows", "fit_seconds"]}
                                | {"heldout_macro_f1": m["evaluation"]["macro_f1"], "training_macro_f1": m["training"]["macro_f1"]})
            if not np.all(seen == 1):
                raise ValueError("OOF coverage incomplete or duplicated")
            pooled = cm_metrics(cm.sum(1))
            boot = cm_metrics(cm[:, draws].sum(2))
            bootstrap[family][arm] = {k:v.mean(0) for k,v in boot.items()}
            summary[family][arm] = {"macro_f1": float(pooled["macro_f1"].mean()),
                "accuracy": float(pooled["accuracy"].mean()),
                "macro_f1_subject_bootstrap_95": np.percentile(boot["macro_f1"].mean(0), [2.5,97.5]).tolist(),
                "train_macro_f1_mean": float(np.mean([f["training"]["macro_f1"] for f in fits])),
                "subject_equal_weighted_macro_f1": float(cm_metrics(cm)["macro_f1"].mean()),
                "per_class_f1": dict(zip(CLASS_NAMES, pooled["per_class_f1"].mean(0).tolist())),
                "records_mean": float(np.mean([f["records"] for f in fits])),
                "windows_mean": float(np.mean([f["windows"] for f in fits])),
                "fit_seconds_mean": float(np.mean([f["fit_seconds"] for f in fits])),
                "confusion_matrix_sum": cm.sum((0,1)).tolist(), "evaluation_kind": "LOSO" if repeats==1 else "repeated_5fold"}
            cubes[f"{family}__{arm}"] = cm
    np.savez_compressed(out / "subject_confusion_matrices.npz", subjects=subjects, **cubes)
    with (out / "fit_metrics.csv").open("w") as f:
        writer=csv.DictWriter(f, fieldnames=list(csv_rows[0])); writer.writeheader(); writer.writerows(csv_rows)
    return p, summary, bootstrap


def diagnostic_plot(out, summary, full):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9.5,5.5))
    x = np.arange(3)
    train = [full["metrics"][f]["training"]["macro_f1"]*100 for f in MODELS]
    test = [summary[f]["subjects_31"]["macro_f1"]*100 for f in MODELS]
    ax.bar(x-.2, train, .38, label="32명 전체 학습자료 재평가 · 성능 검증 아님", color="#9ca3af")
    ax.bar(x+.2, test, .38, label="31명 학습 → 미참여 1명 평가 (32회)", color="#0072B2")
    for xx, val in zip(np.r_[x-.2,x+.2], train+test):
        ax.text(xx,val+1,f"{val:.1f}%",ha="center")
    ax.set(xticks=x, xticklabels=list(NAMES.values()), ylabel="Macro F1 (%)", ylim=(0,114),
           title="높은 점수의 원인: 학습자료 재평가와 새 사람 평가의 차이")
    ax.legend(loc="lower center"); ax.grid(axis="y",alpha=.15)
    fig.tight_layout(); fig.savefig(out / "evaluation_correction.png",dpi=160)
    fig.savefig(out / "evaluation_correction.svg"); plt.close(fig)


def asset_links(out, report):
    folder = report.parent / "assets" / "learning_curve_20260921"
    folder.mkdir(parents=True, exist_ok=True)
    for name in ["learning_curves", "data_acquisition_forecast", "per_class_learning", "evaluation_correction"]:
        for ext in ["png", "svg"]:
            dest = folder / f"{name}.{ext}"
            if dest.is_symlink():
                dest.unlink()
            elif dest.exists():
                raise ValueError("Do not replace an existing real asset")
            dest.symlink_to(os.path.relpath(out / dest.name, folder))
    return "./assets/learning_curve_20260921"


def write_report(out, p, summary, forecasts, gains, full):
    report = Path(p["report"])
    assets = asset_links(out, report)
    rel = os.path.relpath(out, report.parent)
    fullrel = os.path.relpath(Path(p["full_cohort_source"]), report.parent)
    lines = ["# 데이터 규모별 모델 성능과 추가 데이터 수집 전망", "",
        "전체 32명·735개 기록(3초 구간 4,781개)을 같은 기준으로 다시 분석했다. 등록용으로 예약했던 정자세 32개와 라벨 교정 기록 4개를 포함했다. **성능 비교·예측에는 학습에 참여하지 않은 사람의 점수만 사용한다.**", "",
        "[그래프를 포함한 HTML 보고서](./2026-09-21_144323_learning_curve_report.html)", "",
        "## 1. 평가 오류의 원인과 수정 결과", "",
        "이전 7장의 높은 수치는 모델이 학습했던 데이터를 그대로 재평가한 결과였다. 이를 resubstitution(학습자료 재평가)이라고 한다. Random Forest는 깊은 나무로 학습 데이터를 세밀하게 분리하여 학습 F1이 100%가 될 수 있다. 점수 계산 자체의 산술 오류는 아니지만, 데이터 증가에 따른 새 사용자 성능을 알아보려는 목적에 맞는 평가가 아니었다. 서로 다른 평가 기준을 한 보고서에 병치한 구성을 수정했다.", "",
        "수정 방법: 모든 정자세 기록을 일반 라벨 데이터로 포함하되, 동일 사람의 기록은 한 분할에서 전부 학습 또는 전부 평가에만 배치했다. 전체 32명이 평가에 참여하는 LOSO에서는 매번 31명을 학습하고 남은 1명만 평가해 32번의 예측을 합쳤다. 따라서 **32명 데이터로 평가한 성능**이지만 각 평가 모델의 실제 학습 인원은 **31명**이다.", "",
        "| 모델 | 이전 32명 학습자료 F1 (성능 검증 아님) | 수정 후 31→1명 평가 F1 | 평가 Accuracy | 조건부 F1 95% 범위 |",
        "| --- | ---: | ---: | ---: | --- |"]
    for f in MODELS:
        r=summary[f]["subjects_31"]; lo,hi=r["macro_f1_subject_bootstrap_95"]
        lines.append(f"| {NAMES[f]} | {full['metrics'][f]['training']['macro_f1']*100:.2f}% | **{r['macro_f1']*100:.2f}%** | {r['accuracy']*100:.2f}% | {lo*100:.1f}–{hi*100:.1f}% |")
    lines += ["", f"![학습자료 재평가와 미참여자 평가 비교]({assets}/evaluation_correction.png)", "",
        "전체 32명으로 학습한 세 모델은 보존했다. 이 모델들은 모든 사람을 이미 보았으므로 새로운 독립 평가 데이터 없이는 새 사용자 성능을 측정할 수 없다.", "",
        "## 2. 학습 인원과 측정 수에 따른 실측 성능", "",
        "4–25명은 사람 단위 5분할을 두 번 반복했다. 같은 분할에서 인원 순서를 고정해 작은 학습 집합이 큰 집합에 포함되도록 했으며, 세 모델은 같은 데이터를 사용했다. 모든 32명이 각 반복에서 정확히 한 번 평가된다. 31명 행은 평가 방식이 다른 LOSO 보조 결과이며 직접적인 25→31명 증가 효과로 단정하지 않는다.", "",
        "| 학습 인원 | 평균 학습 파일 | 평균 학습 구간 | XGBoost F1 | RBF SVM F1 | Random Forest F1 | 평가 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for n in SIZES+[31]:
        a=f"subjects_{n:02d}"; r=summary["xgboost"][a]
        lines.append(f"| {n} | {r['records_mean']:.1f} | {r['windows_mean']:.1f} | "+" | ".join(f"{summary[f][a]['macro_f1']*100:.2f}%" for f in MODELS)+f" | {'LOSO 32회' if n==31 else '5분할 × 2회'} |")
    lines += ["", f"![등록용 정자세까지 포함한 학습 곡선]({assets}/learning_curves.png)", "",
        "### 같은 25명의 측정 파일 수 증가", "",
        "사람×자세마다 전체 파일을 무작위 순서로 추가했다. 각 층의 최소 1개를 유지하므로 명목 비율과 실제 비율이 다르다. 연속된 구간을 독립 표본처럼 무작위 분할하지 않았다.", "",
        "| 명목 비율 | 실제 파일 비율 | 평균 파일 수 | XGBoost F1 | RBF SVM F1 | Random Forest F1 |", "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    fullcount=summary["xgboost"]["subjects_25"]["records_mean"]
    for nominal, a in [(25,"records_25"),(50,"records_50"),(75,"records_75"),(100,"subjects_25")]:
        count=summary["xgboost"][a]["records_mean"]
        lines.append(f"| {nominal}% | {100*count/fullcount:.1f}% | {count:.1f} | "+" | ".join(f"{summary[f][a]['macro_f1']*100:.2f}%" for f in MODELS)+" |")
    lines += ["", "| 모델 | 20→25명 실측 차이 | 피험자 재표집 차이 범위 | 최소 비율→전체 파일 차이 |", "| --- | ---: | --- | ---: |"]
    for f in MODELS:
        delta=(summary[f]["subjects_25"]["macro_f1"]-summary[f]["subjects_20"]["macro_f1"])*100
        dl,dh=gains[f]["20_to_25_subjects"]
        dr=(summary[f]["subjects_25"]["macro_f1"]-summary[f]["records_25"]["macro_f1"])*100
        lines.append(f"| {NAMES[f]} | {delta:+.2f}%p | {dl:+.2f}–{dh:+.2f}%p | {dr:+.2f}%p |")
    lines += ["", "두 수집 방식의 비용을 맞춘 실험이 아니므로 사람 증가와 반복 측정의 비용 대비 우위를 확정할 수 없다.", "",
        "## 3. 추가 데이터에 따른 전망과 검증", "",
        "등록용 기록을 포함해 다시 얻은 4–25명 미참여자 점수에 inverse-power와 exponential 포화 함수를 맞췄다. 25명 실측값에 두 곡선의 평균 증가량을 더했다. 전체 32명 학습자료 점수는 외삽에 사용하지 않았다. 아래는 같은 모집단·측정 방식·1인당 기록 분포가 유지된다는 조건부 시나리오다.", "",
        "| 모델 | 학습 인원 | 예상 F1 | 25명 대비 | 조건부 시나리오 범위 |", "| --- | ---: | ---: | ---: | --- |"]
    for f in MODELS:
        for e in forecasts[f]["estimates"]:
            lo,hi=e["conditional_scenario_band"]
            lines.append(f"| {NAMES[f]} | {e['training_subjects']} | {e['macro_f1']*100:.1f}% | {e['gain_vs_25_pp']:+.1f}%p | {lo*100:.1f}–{hi*100:.1f}% |")
    lines += ["", f"![측정 파일 수와 추가 인원 전망]({assets}/data_acquisition_forecast.png)", "",
        "피험자 300회 재표집과 두 함수 형태의 2.5–97.5 백분위를 합친 민감도 범위이며, 미래 예측의 95% 보장 구간이 아니다. 단조 개선은 함수의 가정이다. 실제로 개선되지 않거나 성능이 저하될 수 있다. 60명은 곡선 학습 최대 25명의 2배를 넘는다. 표의 인원은 학습 인원이므로 새 독립 평가 인원은 별도로 필요하다.", "",
        "### 작은 규모에서 큰 규모를 예측하는 점검", "",
        "| 모델 | 4–16명으로 20/25명 예측: Power 오차 | Exponential 오차 | 4–25명으로 예측한 31명 F1 | 실제 LOSO F1 |", "| --- | ---: | ---: | ---: | ---: |"]
    loso_checks={}
    for f in MODELS:
        y=[summary[f][f"subjects_{n:02d}"]["macro_f1"] for n in SIZES]
        values=[]
        for kind in ["power","exponential"]:
            curve,_=fit_curve(SIZES,y,kind,[25,31]); values.append(y[-1]+curve[0,1]-curve[0,0])
        pred=float(np.mean(values)); actual=summary[f]["subjects_31"]["macro_f1"]
        loso_checks[f]={"forecast31":pred,"observed_LOSO31":actual,"absolute_error_pp":abs(pred-actual)*100}
        lines.append(f"| {NAMES[f]} | {forecasts[f]['backtest']['power']['mae_pp']:.2f}%p | {forecasts[f]['backtest']['exponential']['mae_pp']:.2f}%p | {pred*100:.2f}% | {actual*100:.2f}% |")
    write_json(out / "loso_forecast_check.json",loso_checks)
    lines += ["", "31명 점수는 곡선 적합 전에 제외해 두고 점검에만 사용했다. 그러나 같은 기존 모집단이고 평가 분할 방식도 달라 완전히 독립적인 미래 예측 검증은 아니다.", "",
        "## 4. 자세별 성능과 과적합", "", f"![자세별 학습 곡선]({assets}/per_class_learning.png)", "",
        "| 자세 | XGBoost LOSO F1 | RBF SVM LOSO F1 | Random Forest LOSO F1 |", "| --- | ---: | ---: | ---: |"]
    for label in CLASS_NAMES:
        lines.append(f"| {label} | "+" | ".join(f"{summary[f]['subjects_31']['per_class_f1'][label]*100:.2f}%" for f in MODELS)+" |")
    lines += ["", "학습 점수와 미참여자 점수의 차이는 과적합과 사람 간 분포 차이를 점검하는 자료다. 높은 학습 점수를 낮추는 것 자체가 수정 목표가 아니며, 평가 자료가 학습에 들어가지 않게 하는 것이 핵심이다. 개인별 지표와 혼동행렬은 수치 결과에 보관했다.", "",
        "## 5. 데이터 구성과 재현 방법", "",
        "전체 32명·735개 기록·4,781개 구간에 등록용 정자세 32개를 포함했다. 4개 라벨 교정은 별도 학습 사본에만 적용했고 원본은 보존했다. 실험 시작 시 확정한 전체 인원 캐시를 사용했으며, 무라벨 측정 전용 자료는 사용하지 않았다. 실행 도중 다른 작업에서 추가한 training_exclusions.json은 이 실험에 소급 적용하지 않았다.", "",
        "| 자세 | 전체 기록 수 | 전체 구간 수 |", "| --- | ---: | ---: |"]
    for i,label in enumerate(CLASS_NAMES):
        lines.append(f"| {label} | {p['cohort']['class_record_counts'][label]} | {p['cohort']['class_window_counts'][i]} |")
    lines += ["", "- 3초/12 Hz/596개 특징, 비중첩 구간. 모든 scaler는 해당 학습 집합에만 fit.",
        "- 기존과 동일한 고정 하이퍼파라미터: XGBoost 200 trees/depth 3/learning rate .05, SVM C=1/gamma=.0005/balanced, Random Forest 200 trees/leaf 1/sqrt/balanced. 확률 argmax를 공통 결정 규칙으로 사용.",
        "- 4–25명 곡선 270개 + 31명 LOSO 96개 = 366개 모델 재학습. 전체 32명 모델 3개는 앞선 전체 학습 실행에서 보존.",
        "- 모든 사람과 그 사람의 정자세 기록을 같은 분할에 배치한다. 임의 구간 분할이나 평가 데이터로 scaler 학습을 하지 않는다.",
        "- 불확실성은 평가 피험자 32명을 통째로 2,000회 재표집. 같은 표본을 모델·규모·반복에 공통 적용. 중복된 학습 집합을 독립 반복으로 간주하지 않는다.",
        "- 기존에 여러 번 조사한 모집단이므로 새로운 외부 최종 검증은 아니다. 물리적 신원·의자·재착석 세션과 센서 동작은 이 연구에서 확인하지 않았다.",
        f"- 실행 시작: {p['started_at']}; 학습 완료: {read(out/'training_complete.json')['finished_at']}; source revision: `{p['source_revision']}`. dirty state/seed/환경/입력 해시는 protocol.json 참조.",
        "- 측정 시각과 학습 시각을 구분하며 파일 mtime으로 센서 측정일을 추정하지 않는다.", "",
        "## 6. 검증·보고서 표시·산출물", "",
        "366개 결과의 모델/예측 해시, 사람·파일 분리, 평가 행의 누락/중복, 정자세 포함 여부를 검사했다. 실제 학습자료 재평가를 `heldout_subjects` 성능으로 집계하는 입력은 거부하도록 했다. 원본 데이터와 운영 모델은 변경하지 않았다.", "",
        "프로젝트 단위 검사 236개와 복구 경로를 포함한 실험 검사 13개가 통과했다. 검사 이후 다른 작업에서 공통 소스가 바뀌어 학습이 자동 중단됐으며, 저장해 둔 소스로 실행 환경을 분리해 같은 캐시·분할·설정으로 재개했다. 이 검사 결과는 검사 당시 소스에 대한 결과이며 이후의 외부 변경까지 검증한 의미는 아니다.",
        f"[소스 변경 감지·복구 기록]({rel}/source_drift.json), [검사 기록]({rel}/verification_tests.json)", "",
        "### VS Code 그래프 표시 수정", "",
        "문서만 열거나 문서 폴더를 작업영역으로 열면 기존의 바깥 research 경로가 Markdown 미리보기의 로컬 리소스 허용 범위에서 벗어날 수 있다. 별도 VS Code 프로필에서 같은 PNG를 두 경로로 열어 기존 경로의 이미지 너비 0, 문서 내부 링크의 정상 너비 2,312를 확인했다.",
        "보고서 폴더 안의 `./assets/learning_curve_20260921/` 링크를 사용한다. 실제 PNG/SVG는 research 실행 폴더에 한 부만 보관하고 문서 폴더에서는 상대 심볼릭 링크로 참조한다. 이미지가 문서 밖의 `../../research/results/` 경로에만 의존하던 구성을 고쳤다. PNG는 기본 미리보기용, SVG는 확대/내보내기용이다.",
        "HTML 보고서는 그래프를 자체 포함하므로 외부 이미지 경로에 의존하지 않는다. VS Code에서는 Markdown 미리보기로 이 문서를 열거나 HTML을 브라우저로 열 수 있다. 미리보기 보안 설정을 낮추지 않는다.", "",
        f"- [실측 수치]({rel}/summary.json), [전체 366개 학습 CSV]({rel}/fit_metrics.csv), [추정값]({rel}/forecast.json), [프로토콜]({rel}/protocol.json)",
        f"- [기존 전체 32명 학습 모델·진단 수치]({fullrel}/summary.json)",
        f"- [VS Code 경로 문제 재현]({rel}/vscode_path_reproduction.json), [최종 미리보기 검증]({rel}/vscode_final_preview.json), [HTML 렌더링 검증]({rel}/html_preview.json)",
        "- [HTML 보고서](./2026-09-21_144323_learning_curve_report.html)", "",
        "저장한 실행의 재개·보고서 재생성 명령이다. 새 실험은 현재 소스에서 원자료 캐시를 다시 감사한 뒤 별도 실행 폴더를 만들어야 한다.",
        "```sh", f".venv/bin/python -m experiments.learning_curve.resume_frozen --output {out.relative_to(ROOT)}",
        f".venv/bin/python -m experiments.learning_curve.integrated_report --output {out.relative_to(ROOT)}", "```", ""]
    if digest(report) != p["report_before_sha256"] and not (out / "report_update.json").exists():
        raise ValueError("Report changed during experiment; merge before replacing")
    report.write_text("\n".join(lines))
    write_json(out / "report_update.json", {"updated_at":now(),"path":str(report),"sha256":digest(report)})
    return report


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--output",type=Path,required=True)
    out=parser.parse_args().output.resolve()
    p,s,b=collect(out); f=forecast(s,b); g=paired_gains(b)
    full=read(Path(p["full_cohort_source"])/"summary.json")
    write_json(out/"summary.json",s); write_json(out/"forecast.json",f); write_json(out/"paired_gains.json",g)
    plot(out,s,f); diagnostic_plot(out,s,full)
    report=write_report(out,p,s,f,g,full)
    env=os.environ.copy()
    bundled_modules=Path.home()/".cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules"
    if not env.get("NODE_PATH") and bundled_modules.is_dir():
        env["NODE_PATH"]=str(bundled_modules)
    renderer=Path(__file__).with_name("render_portable_report.cjs")
    subprocess.run(["node",str(renderer),str(report)],env=env,check=True)
    verify_originals(out)
    source=out/"source_snapshot/analysis"; source.mkdir(exist_ok=True)
    for path in [Path(__file__),Path(__file__).with_name("report.py"),Path(__file__).with_name("resume_frozen.py"),renderer]:
        target=source/path.name
        if not target.exists(): shutil.copy2(path,target)
        elif digest(target)!=digest(path): raise ValueError("Preserve existing analysis snapshot")
    write_json(out/"analysis_complete.json",{"finished_at":now(),"report_sha256":digest(report),"html_sha256":digest(report.with_suffix('.html')),"source_sha256":digest(__file__),"fits_verified":366})
    print(report)


if __name__=="__main__": main()
