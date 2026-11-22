"""Train the feature-based RBF-SVM posture classifier from CSV recordings."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from posture_classifier.data import LABEL_SUSPECT_MARKER, load_dataset
from posture_classifier.evaluation import evaluate_leave_one_subject_out
from posture_classifier.features import FeatureConfig, build_feature_dataset
from posture_classifier.model import PostureSVM, SVMConfig


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("archive"), help="Root directory containing subject CSV folders")
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("outputs/posture_rbf_svm.joblib"),
        help="Destination for the fitted joblib model",
    )
    parser.add_argument("--report-out", type=Path, default=None, help="Destination for the LOSO evaluation JSON report")
    parser.add_argument("--window-size", type=int, default=12, help="Frames per feature window (default: 12)")
    parser.add_argument("--stride", type=int, default=12, help="Frames between adjacent windows (default: 12)")
    parser.add_argument("--c", type=float, default=5.0, help="RBF-SVM C value")
    parser.add_argument("--gamma", default="scale", help="RBF-SVM gamma value or 'scale'/'auto'")
    parser.add_argument("--skip-evaluation", action="store_true", help="Skip Leave-One-Subject-Out evaluation")
    parser.add_argument(
        "--include-label-suspect",
        action="store_true",
        help=f"Include files whose title contains {LABEL_SUSPECT_MARKER!r}; excluded by default",
    )
    return parser.parse_args()


def parse_gamma(value: str) -> str | float:
    if value in {"scale", "auto"}:
        return value
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("gamma must be 'scale', 'auto', or a positive number") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("numeric gamma must be positive")
    return parsed


def main() -> None:
    args = parse_arguments()
    feature_config = FeatureConfig(window_size=args.window_size, stride=args.stride)
    svm_config = SVMConfig(c=args.c, gamma=parse_gamma(args.gamma))
    recordings = load_dataset(args.data_dir, include_label_suspect=args.include_label_suspect)
    feature_dataset = build_feature_dataset(recordings, feature_config)

    print(
        f"Loaded {len(recordings)} recordings from {len(set(feature_dataset.subjects))} subjects "
        f"and built {len(feature_dataset.features)} feature windows."
    )
    if not args.include_label_suspect:
        excluded_count = sum(1 for path in args.data_dir.rglob("*.csv") if LABEL_SUSPECT_MARKER in path.stem)
        print(f"Excluded {excluded_count} label-suspect recording(s).")

    report: dict[str, object] | None = None
    if not args.skip_evaluation:
        evaluation = evaluate_leave_one_subject_out(
            recordings,
            feature_config=feature_config,
            svm_config=svm_config,
        )
        report = evaluation.to_dict()
        print(f"LOSO accuracy: {evaluation.accuracy:.3f}")
        print(f"LOSO macro F1: {evaluation.macro_f1:.3f}")
        print("Confusion matrix rows=true, columns=predicted")
        print("labels:", ", ".join(evaluation.labels))
        print(evaluation.confusion_matrix)

    model = PostureSVM(svm_config).fit(feature_dataset.features, feature_dataset.labels)
    model.save(args.model_out)
    print(f"Saved fitted model to {args.model_out}")

    if report is not None:
        report_path = args.report_out or args.model_out.with_suffix(".metrics.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report.update(
            {
                "recording_count": len(recordings),
                "window_count": int(len(feature_dataset.features)),
                "feature_config": asdict(feature_config),
                "svm_config": asdict(svm_config),
            }
        )
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Saved evaluation report to {report_path}")


if __name__ == "__main__":
    main()
