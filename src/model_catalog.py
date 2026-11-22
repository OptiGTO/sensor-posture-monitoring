"""Training choices and input contracts, without importing optional runtimes."""

from importlib.util import find_spec
import os

# The CPU training contract uses one thread. Initialize OpenMP before optional
# Torch/XGBoost libraries load; late mixed-runtime initialization can crash macOS.
os.environ.setdefault("OMP_NUM_THREADS", "1")

DEFAULT_MODEL_FAMILIES = ("rbf_svm", "logistic_regression", "random_forest", "extra_trees")
MODEL_NAMES = {
    "rbf_svm": "RBF SVM", "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest", "extra_trees": "Extra Trees",
    "catboost": "CatBoost", "minirocket": "MiniRocket + Logistic Regression",
    "cnn1d": "Small 1D-CNN", "xgboost": "XGBoost", "mlp": "Small MLP",
}
# Complete checkpoint recipes are mandatory; these never enter ordinary grids.
CHECKPOINT_MODEL_NAMES = {"equal_soft_voting": "검증된 동일 가중 앙상블",
                          "sagittal_hierarchy": "검증된 자세 계층 분류기",
                          "sequence_soft_voting": "검증된 시계열 동일 가중 앙상블",
                          "record_balanced_extra_trees": "검증된 기록 균형 Extra Trees",
                          "record_bagged_extra_trees": "검증된 기록 배깅 Extra Trees",
                          "record_soft_voting": "검증된 기록 단위 동일 가중 앙상블",
                          "ordinal_extra_trees": "검증된 순서 특징 Extra Trees",
                          "shrinkage_qda": "검증된 축소 공분산 QDA",
                          "spatial_soft_voting": "검증된 공간 특징 동일 가중 앙상블",
                          "record_gain_extra_trees": "검증된 기록별 센서 이득 Extra Trees",
                          "gain_soft_voting": "검증된 센서 이득 동일 가중 앙상블",
                          "record_prototype": "검증된 기록 원형 분류기",
                          "class_conditional_mixture": "검증된 클래스별 혼합 분류기",
                          "local_soft_voting": "검증된 국소 분포 동일 가중 앙상블",
                          "interaction_extra_trees": "검증된 상호작용 특징 Extra Trees",
                          "interaction_soft_voting": "검증된 상호작용 특징 동일 가중 앙상블",
                          "support_extra_trees": "검증된 잎 표본 수 보정 Extra Trees",
                          "support_soft_voting": "검증된 잎 표본 수 보정 동일 가중 앙상블",
                          "tail_extra_trees": "검증된 특징 범위 제한 Extra Trees",
                          "ovr_extra_trees": "검증된 자세별 이진 Extra Trees",
                          "alternatives_soft_voting": "검증된 범위 제한·이진 분류 동일 가중 앙상블",
                          "projection_extra_trees": "검증된 학습 투영 Extra Trees",
                          "projection_soft_voting": "검증된 학습 투영 동일 가중 앙상블",
                          "hardness_extra_trees": "검증된 기록 난이도 Extra Trees",
                          "hardness_random_forest": "검증된 비복원 Random Forest",
                          "hardness_soft_voting": "검증된 기록 난이도 동일 가중 앙상블",
                          "coupled_extra_trees": "검증된 쌍별 확률 결합 Extra Trees",
                          "subject_residual_extra_trees": "검증된 사람 간 성분 제거 Extra Trees",
                          "structured_soft_voting": "검증된 구조화 숲 동일 가중 앙상블"}
SEQUENCE_MODELS = frozenset({"minirocket", "cnn1d", "sequence_soft_voting", "record_gain_extra_trees", "gain_soft_voting",
                             "interaction_extra_trees", "interaction_soft_voting"})
MODEL_PACKAGES = {
    "catboost": {"catboost": ">=1.2"}, "xgboost": {"xgboost": ">=2.0"},
    "minirocket": {"numba": ">=0.63"},
    "cnn1d": {"torch": ">=2.1"}, "mlp": {"torch": ">=2.1"},
}


def missing_model_packages(family: str) -> tuple[str, ...]:
    return tuple(package for package in MODEL_PACKAGES.get(family, {}) if find_spec(package) is None)


def model_display_name(family: str) -> str:
    return MODEL_NAMES.get(family, CHECKPOINT_MODEL_NAMES.get(family, family))


def pipeline_runtime_packages(pipeline: dict) -> dict[str, str]:
    packages = dict(MODEL_PACKAGES.get(pipeline["family"], {}))
    for member in pipeline.get("members", ()):
        packages.update(pipeline_runtime_packages(member))
    return packages


def missing_pipeline_packages(pipeline: dict) -> tuple[str, ...]:
    return tuple(package for package in pipeline_runtime_packages(pipeline) if find_spec(package) is None)


def model_input(family: str, sequences, features=None):
    if family in SEQUENCE_MODELS:
        return sequences
    if features is None:
        from .window_pipeline import extract_window_features
        return extract_window_features(sequences)
    return features


def input_contract(family: str) -> str:
    return "sensor_sequence" if family in SEQUENCE_MODELS else "feature_vector"
