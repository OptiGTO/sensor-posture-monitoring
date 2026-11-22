from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from .classical_models import (
    _optional_catboost,
    _optional_lightgbm,
    _optional_xgboost,
    classification_metrics,
)
from .posture_core import make_subject_pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--input-npz', required=True)
    parser.add_argument('--output-json', required=True)
    parser.add_argument('--split-seed', type=int, default=20260826)
    parser.add_argument('--model-seed', type=int, default=42)
    return parser.parse_args()


def build_model(name: str, seed: int):
    if name == 'XGBoost':
        model = _optional_xgboost(seed)
    elif name == 'LightGBM':
        model = _optional_lightgbm(seed)
    elif name == 'CatBoost':
        model = _optional_catboost(seed)
    elif name == 'Gradient Boosting':
        model = GradientBoostingClassifier(
            n_estimators=120,
            learning_rate=0.05,
            max_depth=2,
            subsample=0.85,
            random_state=seed,
        )
    else:
        raise ValueError(f'지원하지 않는 격리 모델: {name}')
    if model is None:
        raise ImportError(f'{name} 라이브러리가 설치되어 있지 않습니다.')
    return model


def main() -> None:
    args = parse_args()
    data = np.load(args.input_npz, allow_pickle=True)
    X = data['X']
    y = data['y'].astype(int)
    groups = data['groups'].astype(str)
    pairs = make_subject_pairs(groups, seed=args.split_seed)
    start = time.perf_counter()
    rows = []
    for fold_index, test_subjects in enumerate(pairs, start=1):
        test_mask = np.isin(groups, test_subjects)
        model = build_model(args.model, args.model_seed)
        model.fit(X[~test_mask], y[~test_mask])
        prediction = np.asarray(model.predict(X[test_mask])).reshape(-1).astype(int)
        rows.append(
            {
                'model': args.model,
                'fold': fold_index,
                'test_subjects_raw': list(map(str, test_subjects)),
                **classification_metrics(y[test_mask], prediction),
            }
        )
    payload = {
        'model': args.model,
        'rows': rows,
        'training_seconds': time.perf_counter() - start,
    }
    Path(args.output_json).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8'
    )


if __name__ == '__main__':
    main()
