#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference_core import PosturePredictor
from src.posture_core import read_sensor_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3초 또는 5초 창 자세 분류")
    parser.add_argument("--input", required=True, help="센서 CSV")
    parser.add_argument("--window", choices=["3", "5", "both"], default="both")
    parser.add_argument("--model-dir", default=str(PROJECT_ROOT / "artifacts"))
    parser.add_argument("--output", default=None, help="결과 JSON 경로")
    return parser.parse_args()


def predict_one(input_path: Path, model_path: Path) -> dict[str, object]:
    predictor = PosturePredictor(model_path)
    frame = read_sensor_csv(input_path)
    probability, metadata = predictor.predict_windows(frame)
    labels = list(predictor.labels)
    window_results = []
    for row, values in zip(metadata.to_dict(orient="records"), probability):
        predicted_id = int(values.argmax())
        window_results.append(
            {
                "창_번호": int(row["window_index"]),
                "시작_초": float(row["start_s"]),
                "명목상_종료_초": float(row["nominal_end_s"]),
                "예측_자세": labels[predicted_id],
                "예측_클래스_ID": predicted_id,
                "최대_확률": float(values[predicted_id]),
                "클래스별_확률": {label: float(value) for label, value in zip(labels, values)},
            }
        )
    mean_probability = probability.mean(axis=0)
    overall_id = int(mean_probability.argmax())
    return {
        "사용_모델": predictor.model_name,
        "창_길이_초": int(predictor.window_seconds),
        "창_개수": len(window_results),
        "전체_기록_예측_자세": labels[overall_id],
        "전체_기록_예측_클래스_ID": overall_id,
        "전체_기록_최대_평균확률": float(mean_probability[overall_id]),
        "전체_기록_클래스별_평균확률": {
            label: float(value) for label, value in zip(labels, mean_probability)
        },
        "창별_예측": window_results,
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    model_dir = Path(args.model_dir).expanduser().resolve()
    windows = (3, 5) if args.window == "both" else (int(args.window),)
    result = {
        "입력_파일": input_path.name,
        "결과": {
            f"{seconds}초": predict_one(
                input_path, model_dir / f"posture_classifier_window_{seconds}s.joblib"
            )
            for seconds in windows
        },
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
