#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference_core import PosturePredictor  # noqa: E402
from src.posture_core import read_sensor_csv  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="새 센서 CSV에서 7개 자세 중 하나를 예측합니다."
    )
    parser.add_argument("--input", required=True, help="예측할 센서 CSV 파일")
    parser.add_argument(
        "--model",
        default=str(PROJECT_ROOT / "artifacts" / "posture_classifier.joblib"),
        help="학습된 joblib 모델 번들",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="예측 결과를 저장할 JSON 파일(미지정 시 화면 출력만 수행)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    predictor = PosturePredictor(model_path)

    frame = read_sensor_csv(input_path)
    probability = predictor.predict_frame(frame)
    predicted_id = int(probability.argmax())
    labels = list(predictor.labels)
    result = {
        "입력_파일": input_path.name,
        "사용_모델": predictor.model_name,
        "예측_자세": labels[predicted_id],
        "예측_클래스_ID": predicted_id,
        "최대_확률": float(probability[predicted_id]),
        "클래스별_확률": {
            label: float(value) for label, value in zip(labels, probability)
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
