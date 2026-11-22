"""Write an immutable Korean report after an iterative evaluation is sealed."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
from statistics import mean


ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def percentage(value):
    return f"{float(value) * 100:.2f}%"


def delta(value):
    return f"{float(value) * 100:+.2f}%p"


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def table(headings, rows):
    return ["| " + " | ".join(map(cell, headings)) + " |",
        "| " + " | ".join("---" for _ in headings) + " |",
        *["| " + " | ".join(map(cell, row)) + " |" for row in rows], ""]


def code(value, language="json"):
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) if language == "json" else str(value)
    return [f"```{language}", text.rstrip(), "```", ""]


def link(report, target, label):
    return f"[{label}]({Path(os.path.relpath(target, report.parent)).as_posix()})"


def recipe_description(spec):
    if spec["family"] in {"equal_soft_voting", "sequence_soft_voting", "record_soft_voting", "spatial_soft_voting", "gain_soft_voting", "local_soft_voting", "interaction_soft_voting", "support_soft_voting", "alternatives_soft_voting", "projection_soft_voting", "hardness_soft_voting", "structured_soft_voting"}:
        return f"개별 통과 후보 {len(spec['members'])}개의 클래스 정렬 확률을 동일 가중 평균"
    if spec["family"] == "coupled_extra_trees":
        return (f"공유 TRAIN StandardScaler·원596개 특징; 사전순21개 자세 쌍별 클래스 균형 Extra Trees head당 {spec['model_params']['n_estimators']}개; "
            "seed=(s+pair_index)%2**32; r_ij=P(i|i,j)를 float64·1e-7 제한 후 Wu 방식2의8×8 equality-KKT 직접 풀이로7클래스 결합; argmax·확률 보정 주장 없음")
    if spec["family"] == "subject_residual_extra_trees":
        change = ("원 표준화 ndarray를 그대로 전달하는 항등 경로" if spec["feature_engineering"]["mode"] == "identity" else
            "TRAIN 기록→사람×자세→사람 균등 평균의 중심화된 사람 중심점 행렬B; full SVD 상위3방향V; z-(zVᵀ)V")
        return "원596개 특징·비가중 TRAIN StandardScaler; " + change + "; 재표준화 없이 클래스 균형 Extra Trees·원 확률 argmax"
    if spec["family"] in {"hardness_extra_trees", "hardness_random_forest"}:
        coefficient = spec["training"]["hardness"]["coefficient"]
        weighting = ("OOF·추가 sample_weight 없이 기존 클래스 균형 학습" if coefficient == 0 else
            "TRAIN 내부 사람별 GroupKFold4; fold마다 독립 StandardScaler·기준 ET; 기록별 OOF 오분류율 e로 h=1+0.5e, 클래스 안의 창 평균으로 정규화한 sample_weight")
        forest = "Random Forest(bootstrap=False, best splitter)" if spec["family"] == "hardness_random_forest" else "Extra Trees(random splitter)"
        return "기존596개 특징·전체 TRAIN의 비가중 StandardScaler; " + weighting + "; 클래스 균형 " + forest + "·클래스 정렬 확률 argmax"
    if spec["family"] == "projection_extra_trees":
        change = {"identity": "원 표준화 특징을 그대로 전달하는 무변형 대조군",
            "pca64": "TRAIN float64 PCA64(full SVD, whiten=False); 평균 중심화 후 저장된 64개 축으로 투영",
            "all_plus_lda6": "TRAIN float64 LDA6(eigen, 자동 수축, 경험적 클래스 사전확률); 상수 열 제외·축 부호 고정; 원 표준화596+LDA6=602개 좌표"}[spec["feature_engineering"]["mode"]]
        return "기존596개 특징의 TRAIN StandardScaler → " + change + "; 클래스 균형 Extra Trees·원래 확률 argmax; 투영 뒤 재표준화 없음"
    if spec["family"] == "tail_extra_trees":
        fraction = spec["preprocessing"]["tail_fraction"]
        clipping = ("클리핑 없이 원 입력 그대로" if fraction == 0 else
            f"TRAIN에서 float64 linear 분위수 {fraction:g}·{1-fraction:g} 경계를 구해 특징별 제한; 원 입력 자료형으로 복원")
        return clipping + "; 기존 596개 특징·제한 후 TRAIN StandardScaler·클래스 균형 Extra Trees·클래스 정렬 argmax"
    if spec["family"] == "ovr_extra_trees":
        return (f"기존 596개 특징·공유 TRAIN StandardScaler; 7개 이진 Extra Trees head당 {spec['model_params']['n_estimators']}개 트리; "
            "각 head 양성/음성 클래스 균형·seed=(s+c)%2**32; 양성 점수 합 정규화, 정확히 0인 행은 균등분포·고정 클래스 순서 argmax")
    if spec["family"] == "support_extra_trees":
        alpha = spec["postprocessing"]["alpha"]
        scoring = ("기존 forest.predict_proba 직접 호출" if alpha == 0 else
            f"트리별 q=(n×p+{alpha}/7)/(n+{alpha}), n은 해당 잎의 비가중 훈련 창 수; 고정 트리 순서로 q 평균")
        return (f"기존 596개 특징·훈련 StandardScaler·클래스 균형 Extra Trees {spec['model_params']['n_estimators']}개; "
            + scoring + "; 클래스 정렬 후 argmax")
    if spec["family"] == "interaction_extra_trees":
        mode = spec["feature_engineering"]["mode"]
        transformation = {"identity": "동일 시퀀스에서 기존 596개 특징을 재추출하는 무변형 대조군",
            "hadamard512": "기존 596개 특징의 훈련 StandardScaler 뒤에서 센서 통계 블록 512개 좌표에 고정 H16/4 변환; 나머지 84개 유지; 재표준화 없음",
            "temporal_corr120": "기존 596개 특징에 같은 창의 모든 센서 쌍 Pearson 시간 상관 120개 추가; 총 716개를 훈련 StandardScaler로 변환"}[mode]
        return transformation + "; 클래스 균형 Extra Trees·고정 200개 트리·깊이8·잎최소8·특징비율0.5·확률 argmax"
    if spec["family"] == "record_prototype":
        return ("기존 596개 특징·훈련 창 StandardScaler·기록별 평균 대표점; "
            f"각 클래스의 가장 가까운 대표점 {spec['model_params']['n_neighbors']}개의 Euclidean 거리 평균; "
            "1/(거리+1e-12) 점수를 7개 클래스에 정규화 후 argmax; 평가 창은 독립적으로 예측")
    if spec["family"] == "class_conditional_mixture":
        return ("기존 596개 특징·훈련 창 StandardScaler·각 클래스별 대각 공분산 GaussianMixture; "
            f"성분 {spec['model_params']['n_components']}개·reg_covar=0.1·kmeans 초기화 3회·최대 200회·tol=0.001; "
            "동일 클래스 사전확률·안정화한 로그밀도 정규화 후 argmax")
    if spec["family"] == "record_gain_extra_trees":
        augmentation = spec["training"]["augmentation"]
        direction = {"identity": "배율 1 대조군", "common": "기록별 모든 센서에 공통 배율", "per_sensor": "기록별 센서마다 별도 배율"}[augmentation["mode"]]
        return (f"{direction}; 균등분포 [1-{augmentation['strength']}, 1+{augmentation['strength']}]·기록 해시와 시드로 고정; "
            "훈련 시퀀스만 교체하고 창 수 유지; 기존 596개 특징 재추출·훈련 StandardScaler·클래스 균형 Extra Trees; 추론은 증강 없이 특징 추출·확률 argmax")
    if spec["family"] == "ordinal_extra_trees":
        count = 240 if spec["feature_engineering"]["mode"] == "all_plus_mean_median_order" else 120
        return (f"기존 596개 특징 보존 + 절대압력 센서 평균의 모든 120쌍 순서 비교"
            + (" 및 중앙값의 120쌍 순서 비교" if count == 240 else "")
            + f"; 총 {596 + count}개 특징·훈련 StandardScaler·클래스 균형 Extra Trees·확률 argmax")
    if spec["family"] == "shrinkage_qda":
        return ("사전 지정 절대·상대 센서 평균 32개·훈련 StandardScaler; 클래스별 공분산 QDA; "
            f"reg_param={spec['model_params']['reg_param']}로 각 클래스 공분산 고유값을 단위값으로 수축; "
            "SVD solver·7개 클래스 동일 사전확률·tol=1e-4·공분산 저장·확률 argmax")
    if spec["family"] == "record_balanced_extra_trees":
        return ("기존 596개 특징·훈련 StandardScaler; 같은 클래스의 각 기록에 동일한 총 가중치, 각 클래스에도 동일한 총 가중치; "
            "창 가중치 N/(7×클래스 기록 수×해당 기록 창 수); Extra Trees class_weight=None으로 중복 가중 방지; 확률 argmax")
    if spec["family"] == "record_bagged_extra_trees":
        return ("기존 596개 특징·전체 훈련 행 StandardScaler; 클래스별 기록 ID 복원추출 후 뽑힌 기록의 모든 창을 유지; "
            f"총 {spec['model_params']['n_estimators']}개 트리를 {spec['model_params']['bag_count']}개 Extra Trees에 균등 배분; "
            "각 묶음에서 클래스 균형 가중치; 클래스 정렬 확률 동일 평균 후 argmax")
    if spec["family"] == "cnn1d":
        return ("36×16 센서 시계열; 훈련 사람의 센서별 평균·표준편차 정규화; 1D CNN·클래스 균형 교차엔트로피·AdamW; "
            "고정 epoch 중 최소 훈련 손실 상태 저장, 외부 검증·patience 조기 종료 미사용; softmax argmax")
    if spec["family"] == "minirocket":
        return ("36×16 센서 시계열; 훈련 데이터로만 MiniRocket 다변량 임계값 적합; 변환 특징 StandardScaler; "
            "클래스 균형 LogisticRegression(max_iter=2000) 확률 argmax")
    if spec["family"] == "sagittal_hierarchy":
        return ("기존 596개 특징·학습 StandardScaler; 동일 모델군의 5갈래 상위 분류와 정자세·앞·뒤 3갈래 전문 분류; "
            "상위 학습 가중치는 원래 7개 자세 빈도로 계산; 최종 확률은 조건부 확률의 곱으로 결합")
    mode = spec["feature_engineering"]["mode"]
    feature = {
        "all": "절대·상대압력 통계 596개 유지",
        "static": "센서 평균·표준편차·중앙값·최솟값·최댓값·범위·IQR·RMS와 공간 집계 평균·표준편차 312개",
        "means": "센서 평균·중앙값과 행·열·공간차 평균 92개",
        "relative_static": "프레임 상대 표준화 신호의 정적 통계 156개",
        "contrast": "창 내부 오프셋·크기 변동을 제거한 압력 분포·질량중심·시간 변동 75개",
        "all_plus_contrast": "기존 596개와 창 내부 압력 분포 대비 75개를 결합한 671개",
        "subject_invariant": f"학습 자세 F / (1 + {spec['feature_engineering'].get('strength')} × 학습 사람 F) 순위 상위 {spec['feature_engineering'].get('k')}개 특징",
        "shrinkage_lda": f"훈련 StandardScaler·상수 특징 제거·수축 LDA로 자세 판별 축 {spec['feature_engineering'].get('n_components') or '최대 6'}개 생성",
    }[mode]
    weighting = ("사람별 총 가중치 및 각 사람 안의 관측 자세별 가중치 균등화; scaler에도 동일 가중치"
        if spec["training"]["weighting"] == "subject_class_balanced" else "클래스 균형 가중치")
    scaler = ("훈련 StandardScaler 후 Ledoit–Wolf 수축 공분산 whitening"
        if spec["preprocessing"]["scaler"] == "shrinkage_whiten" else
        "학습 창 전체의 최대 256 분위수를 정규분포로 변환; 평가 범위 밖 값은 훈련 분위수 끝점에 제한"
        if spec["preprocessing"]["scaler"] == "quantile_normal" else "훈련 StandardScaler")
    output = ("SVC OVR 결정점수 softmax 후 argmax; 보정된 확률을 의미하지 않음"
        if spec["postprocessing"]["kind"] == "decision_softmax" else
        "훈련 피험자 내부 GroupKFold OOF 예측으로 제한된 자세별 log bias를 학습한 판정점수 argmax; 보정 확률 아님"
        if spec["postprocessing"]["kind"] == "macro_f1_bias" else "클래스 정렬 predict_proba argmax")
    return f"{spec['family']}; {feature}; {scaler}; {weighting}; {output}"


def hypothesis(spec, *, identity_control=False):
    if identity_control:
        return ("무변형 경로 검증: 기준과 같은 학습 행·설정으로 모든 개발 예측·확률·지표를 완전히 재현하는지 확인한다. "
            "사전 등록한 소프트웨어 대조군이며 성능 개선 후보가 아니다.")
    if spec["family"] in {"equal_soft_voting", "sequence_soft_voting", "record_soft_voting", "spatial_soft_voting", "gain_soft_voting", "local_soft_voting", "interaction_soft_voting", "support_soft_voting", "alternatives_soft_voting", "projection_soft_voting", "hardness_soft_voting", "structured_soft_voting"}:
        return "개별 성공 후보의 서로 다른 오류를 확률 평균으로 줄인다. 실패 후보는 결합에 포함하지 않는다."
    if spec["family"] == "coupled_extra_trees":
        return ("각 자세 쌍에 특화된 이진 분할과 상호 일관성 목적함수로 다중분류의 지속적인 혼동을 줄일 수 있는지 확인한다. "
            "기존 OVR·계층분류와 다르며 head당32개·깊이8·잎8을 평가 전에 고정한다. 결합은 확률 calibration을 보장하지 않는다.")
    if spec["family"] == "subject_residual_extra_trees":
        return ("TRAIN 사람 중심점의 상위3방향을 제거하면 평가 사람의 ID·교정 없이 일반화가 개선되는지 확인한다. "
            "사람 간 차이가 순수 방해 정보라는 전제는 두지 않으며 유용한 자세 정보 손실 위험도 평가한다. 차원3은 사전 고정한다.")
    if spec["family"] == "hardness_random_forest":
        return ("같은 원 특징·학습 행·트리 수·깊이·잎 크기에서 bootstrap=False Random Forest의 best 분할이 "
            "Extra Trees의 무작위 임곗값 분할보다 유리한지 비교한다. 기존 RF의 bootstrap=True 설정과 구분하며 새로운 아키텍처라고 주장하지 않는다.")
    if spec["family"] == "hardness_extra_trees":
        return ("외부 평가 사람을 사용하지 않고 TRAIN 내부에서 얻은 기록별 OOF 오류율로 어려운 기록의 기여를 제한적으로 높이면 "
            "지속적인 자세 혼동을 줄이는지 확인한다. 0.5 강도는 사전 고정하며 높은 오류가 라벨 오류인지 실제 변이인지는 알 수 없다.")
    if spec["family"] == "projection_extra_trees":
        return {"identity": "학습 투영을 끈 처리 경로가 기존 기준의 모든 개발 예측과 지표를 정확히 재현해야 한다. 성능 선택 대상이 아닌 소프트웨어 대조군이다.",
            "pca64": "라벨을 보지 않는 TRAIN 공분산의 64개 축이 중복된 표현을 줄이고 다른 트리 분할 방향을 제공하는지 확인한다. 차원은 평가 전에 고정하며 낮은 분산의 자세 정보도 사라질 수 있다.",
            "all_plus_lda6": "기존 596개 축을 유지하면서 TRAIN 자세 구분 방향6개를 추가하면 깊이가 제한된 트리의 표현을 보완할 수 있는지 확인한다. 005의 LDA6-only 실패 원인을 확정하거나 실패 후보를 성공 조합으로 재사용하는 것이 아니다."}[spec["feature_engineering"]["mode"]]
    if spec["family"] == "tail_extra_trees":
        return (f"훈련 특징의 양쪽 {100 * spec['preprocessing']['tail_fraction']:g}% 꼬리 값을 저장된 경계로 제한하면 "
            "무작위 분할이 유용한 범위에 더 많이 놓이는지 검증한다. 실제 자세 구분에 필요한 값도 제한될 수 있으며 데이터 오류를 전제하지 않는다.")
    if spec["family"] == "ovr_extra_trees":
        return ("일곱 개의 독립적인 자세별 이진 분할 목적함수가 다중분류 숲의 반복된 자세 혼동을 줄이는지 검증한다. "
            "전체 트리 수와 클래스 가중치도 달라지므로 분해 구조만의 효과로 단정하지 않는다.")
    if spec["family"] == "support_extra_trees":
        alpha = spec["postprocessing"]["alpha"]
        if alpha == 0:
            return "같은 훈련 행·특징·깊이·최소 잎 크기에서 트리 수만 늘리면 무작위 트리 생성에 따른 변동이 줄어드는지 확인한다."
        return (f"완화 강도 {alpha}로 작은 잎의 클래스 점수를 동일 사전 점수에 가깝게 만들면 적은 훈련 창에 대한 확신의 영향을 줄일 수 있는지 확인한다. "
            "학습된 트리·표준화·분할은 바꾸지 않는다. 겹치는 창 수는 독립 피험자 수가 아니며 결과를 보정된 사후확률로 해석하지 않는다.")
    if spec["family"] == "interaction_extra_trees":
        return {"identity": "새 시퀀스→특징 추출→표준화→Extra Trees 경로가 기존 특징 캐시 기반 기준의 모든 개발 예측과 지표를 완전히 재현해야 한다. 소프트웨어 대조군이며 개선 후보가 아니다.",
            "hadamard512": "정규화된 센서 통계의 고정 선형 조합을 트리의 좌표축으로 사용하면 같은 깊이에서도 다른 상호작용 경계를 표현할 수 있는지 확인한다. 차원을 줄이거나 훈련 공분산을 학습하지 않으며 물리적 센서 회전 불변성을 뜻하지 않는다.",
            "temporal_corr120": "단일 센서 통계에 명시적으로 없는 센서 쌍의 동시 변동을 추가하면 지속적인 자세 혼동을 줄일 수 있는지 확인한다. 모든 120쌍을 사전에 고정하고 해당 창 내부에서만 계산한다. 특징 수 증가로 트리가 검토하는 분할 후보 수도 달라진다."}[spec["feature_engineering"]["mode"]]
    if spec["family"] == "record_prototype":
        return ("같은 기록의 상관된 창을 훈련 평균 대표점으로 압축하고 각 클래스의 가까운 기록과 비교하면 지속적인 자세 혼동이 줄어드는지 확인한다. "
            "일반적인 창 단위 KNN과 다른 가설이며, 각 클래스의 가장 가까운 1개 또는 3개 기록을 사용한다. "
            "평가 기록을 평균하거나 사람별로 적응하지 않는다. 기록이 많은 사람의 대표점 수는 여전히 많다.")
    if spec["family"] == "class_conditional_mixture":
        return ("자세마다 하나의 평균만 사용하는 대신 두 개의 훈련 분포 성분을 허용하면 서로 다른 압력 양상을 표현할 수 있는지 확인한다. "
            "성분 1개와 2개의 동일한 대각 공분산 설정을 비교한다. 기준 Extra Trees와의 차이에는 분류기 가정 변경이 포함된다. "
            "성분은 사람 정체성이 아닌 자세 구조를 표현한다는 보장이 없으며 평가 라벨로 성분 수나 정규화를 맞추지 않는다.")
    if spec["family"] == "record_gain_extra_trees":
        mode = spec["training"]["augmentation"]["mode"]
        if mode == "identity":
            return "배율 1로 같은 시퀀스→596개 특징→Extra Trees 경로가 기존 캐시 기반 기준의 예측을 완전히 재현하는지 확인한다. 소프트웨어 경로 대조군이며 성능 개선 가설이 아니다."
        change = "기록 전체의 센서 크기" if mode == "common" else "기록 안의 센서 간 상대 크기"
        return (f"{change}에 작은 무작위 변동을 주면 특정 기록 수치에 대한 과적합이 줄어드는지 검증한다. "
            "같은 기록의 모든 창·시점에 같은 양의 배율을 적용한다. 원 창 수를 유지하고 특징을 센서 공간에서 다시 추출한다. "
            "배율은 수치적 규제 가설이며 실제 센서 교정 오차를 측정해 정한 범위가 아니다.")
    if spec["family"] == "ordinal_extra_trees":
        return ("센서 두 위치의 크기 순서를 직접 특징으로 제공하면 단일 특징 임계값을 사용하는 트리가 공간 관계를 더 쉽게 학습할 수 있다. "
            "모든 센서 쌍을 사전에 고정하고 원래 596개 특징을 유지한다. 추가 순서 특징은 공통 오프셋·양의 이득 변화에 불변이나 원래 특징은 그렇지 않다. "
            "max_features 비율 0.5는 같아도 전체 차원 증가로 각 분할이 검토하는 특징 수는 달라진다.")
    if spec["family"] == "shrinkage_qda":
        return ("32개 센서 평균의 클래스별 상관 구조가 지속적인 자세 혼동을 줄이는지 검증한다. "
            "이전 공유 공분산 LDA 특징 축소와 달리 클래스별 공분산으로 직접 분류하며, 강한 수축으로 작은 코호트의 추정 변동을 제한한다. "
            "특징·수축 강도·동일 클래스 사전확률은 평가 전에 고정한다.")
    if spec["family"] == "record_balanced_extra_trees":
        return ("사람당 기록 수의 정보는 유지하면서 한 기록이 6개 또는 7개 창으로 나뉘는 데 따른 기여 차이를 줄인다. "
            "실패했던 사람 균등 가중치와 구분하며, 같은 자세의 각 기록에 동일한 총 학습 가중치를 부여한다. 가중치는 훈련 기록으로만 계산한다.")
    if spec["family"] == "record_bagged_extra_trees":
        return ("한 기록 안의 상관된 창을 함께 재표집하여 기록 구성에 따른 모델 변동을 평균한다. "
            "전체 트리 수는 기준과 같은 200개로 고정하고, 훈련 기록만 클래스별로 복원추출한다. 평가 창과 평가 기록은 바꾸지 않는다.")
    if spec["family"] in {"cnn1d", "minirocket"}:
        return ("동일한 3초 창의 시간 순서를 유지해 요약 통계에서 사라진 압력 변화 패턴을 학습한다. "
            + ("작은 CNN과 강한 dropout·가중치 감쇠로 사람별 과적합을 제한한다." if spec["family"] == "cnn1d" else
               "고정 무작위 합성곱 변환과 규제된 선형 판독기로 시간 구조를 비교한다.")
            + " 기준과 동일하게 학습 피험자의 일반 정자세 기록을 복원하며 평가 창은 바꾸지 않는다.")
    if spec["family"] == "sagittal_hierarchy":
        return "정자세·앞·뒤를 하나로 묶은 상위 분류와 이 셋만의 전문 분류를 새로 학습하면 지속적인 내부 혼동을 줄일 수 있다; 딱 하나의 갈래로 강제 이동하지 않고 모든 자세 확률을 유지한다."
    mode = spec["feature_engineering"]["mode"]
    parts = []
    if spec["family"] == "extra_trees":
        params = spec["model_params"]
        if params.get("criterion") == "entropy":
            parts.append("트리 분할의 Gini 목적함수를 정보 엔트로피로 바꿔 다중 자세 분포의 분리 기준을 비교한다; 특징·학습 기록·트리 수는 유지한다")
        if params.get("ccp_alpha", 0) > 0:
            parts.append("각 트리를 먼저 학습한 뒤 비용-복잡도 가지치기로 작은 잎의 복잡도를 줄여 사람별 우연한 분할의 영향을 낮춘다; alpha는 개발 점수를 보기 전에 고정한다")
        if params.get("min_impurity_decrease", 0) > 0:
            parts.append("표본 가중 불순도 감소가 작은 분할을 학습 중 차단하여 약한 압력 차이에 대한 과적합을 줄인다; 기존 깊이·최소 잎 크기는 유지한다")
    if spec["family"] == "mlp":
        parts.append("작은 비선형 신경망과 dropout·AdamW 규제로 다른 특징 상호작용을 학습한다; 외부 검증을 epoch 선택에 사용하지 않고 고정 횟수 학습 중 최소 학습 손실 상태를 저장한다")
    if mode == "subject_invariant":
        parts.append("학습 데이터에서 자세와 관련되면서 사람 구분에는 덜 민감한 특징을 골라 새로운 사람의 일반화를 개선한다; F 값은 순위 점수이며 독립 표본 유의확률이 아니다")
    elif mode == "shrinkage_lda":
        parts.append("훈련 데이터로만 수축 공분산 판별 축을 학습하여 중복 특징을 압축하고 자세 관련 구조에 집중한다; 이후 분류기의 비선형 결정 경계를 평가한다")
    elif mode in {"contrast", "all_plus_contrast"}:
        parts.append("창 내부의 공통 오프셋과 양의 크기 배율에 덜 의존하는 압력 분포를 사용해 사람 간 변동을 줄인다")
    elif mode != "all":
        parts.append("짧은 3초 창의 주파수·고차 통계 변동을 줄이고 압력 분포 요약에 집중한다")
    if mode == "relative_static":
        parts.append("절대 크기 의존을 낮춘다")
    if spec["training"]["weighting"] == "subject_class_balanced":
        parts.append("기록이 많은 피험자의 영향과 피험자 안의 클래스 불균형을 줄인다")
    if spec["preprocessing"]["scaler"] == "shrinkage_whiten":
        parts.append("상관이 높은 특징의 중복 기여를 완화해 SVM 거리 구조를 개선한다")
    if spec["preprocessing"]["scaler"] == "quantile_normal":
        parts.append("훈련 분위수를 이용한 비선형 정규분포 변환으로 극단 압력 통계의 영향을 제한하고 랜덤 트리 분할·SVM 거리 구조를 바꾼다")
    if spec["postprocessing"]["kind"] == "decision_softmax":
        parts.append("SVC 내부 확률 보정에 따른 클래스 순위 변화를 제거한 결정점수 순위가 유리한지 확인한다")
    if spec["postprocessing"]["kind"] == "macro_f1_bias":
        parts.append("외부 검증·고정 평가를 보지 않고 학습 피험자 내부 교차 예측으로 클래스별 판정 편향을 조정하면 지속적인 자세 혼동이 줄어드는지 확인한다; 내부 탐색의 과적합 위험을 외부 개발 분할에서 평가한다")
    if spec["training"].get("data_policy") == "include_ordinary_reserved_training_records_v1":
        parts.append("과거 임의 예약한 일반 정자세 기록을 학습 피험자에 한해 복원하면 정자세 표현의 학습 범위가 넓어진다; 실제 등록·교정 기록은 제외한다")
    return "; ".join(parts) + "." if parts else "기존 전체 특징 모델을 같은 입력·분할·시드에서 다시 학습해 모델군 차이를 확인한다."


def verified_fit(out, directory, protocol_hash):
    row = read(directory / "metrics.json")
    if read(directory / "metrics_seal.json")["sha256"] != digest(directory / "metrics.json"):
        raise ValueError(f"Fit metrics seal changed: {directory}")
    if row["protocol_sha256"] != protocol_hash:
        raise ValueError(f"Fit protocol differs: {directory}")
    for filename, key in (("model.joblib", "model_sha256"), ("predictions.npz", "predictions_sha256")):
        if digest(directory / filename) != row[key]:
            raise ValueError(f"Fit artifact changed: {directory / filename}")
    return row


def next_experiments(candidates):
    recipes = [row["pipeline"] for row in candidates]
    if any(spec["family"] in {"coupled_extra_trees", "subject_residual_extra_trees", "structured_soft_voting"} for spec in recipes):
        return ["1. 개별 개발 기준 통과 후보만 결합·추가 시드·고정 평가·재학습 검증으로 진행합니다.",
            "2. 실패한 pair head 수나 제거 차원을 개발 점수에 맞춰 재탐색하지 않습니다. TRAIN 중심점에서 제거된 자세 정보·DEV 사람 및 클래스별 변화로 독립 가설을 검토합니다.",
            "3. 신규 피험자·재착석 세션의 전향적 검증이 필요합니다. 기존 반복 평가 점수만으로 현장 개선을 주장하지 않습니다."]
    if any(spec["family"] in {"hardness_extra_trees", "hardness_random_forest", "hardness_soft_voting"} for spec in recipes):
        return ["1. 개별 개발 기준을 통과한 분할·가중치 구성만 호환 결합 및 추가 시드·고정 평가로 진행합니다.",
            "2. 미달한 가중치 강도나 inner fold 수를 개발 점수에 맞춰 재탐색하지 않습니다. 저장된 TRAIN OOF·기록별 가중치와 클래스/사람별 변화에서 다음 독립 가설의 근거를 확인합니다.",
            "3. 신규 피험자·재착석 세션으로 전향적 평가를 준비합니다. 같은 개발 분할을 반복 사용한 성과와 물리적 센서 검증은 구분합니다."]
    if any(spec["family"] in {"projection_extra_trees", "projection_soft_voting"} for spec in recipes):
        return ["1. 개별 투영 구성이 개발 기준을 통과한 경우에만 추가 시드와 호환 결합을 검증합니다.",
            "2. 실패하면 PCA 차원이나 LDA 축을 개발 점수에 맞춰 재탐색하지 않습니다. 저장된 TRAIN 투영·클래스 혼동과 기존 독립 가설의 중복을 검토합니다.",
            "3. 새로운 피험자·재착석 세션 자료를 확보해 반복 개발 평가의 한계를 확인합니다. 현재 결과만으로 물리적 센서 강건성을 주장하지 않습니다."]
    if any(spec["family"] in {"tail_extra_trees", "ovr_extra_trees", "alternatives_soft_voting"} for spec in recipes):
        return ["1. 개별 개발 기준을 통과한 범위 제한·이진 분류 구성만 추가 시드와 호환 결합으로 검증합니다.",
            "2. 범위 제한이 실패하면 분위수를 개발 점수에 맞춰 계속 조정하지 않습니다. 기존 TRAIN의 클래스별 극단값 정보와 노드별 범위를 별도 진단한 뒤 새로운 가설을 검토합니다.",
            "3. 이진 분류가 성공하면 실제 시간·저장 크기·추론 비용을 측정하고 용량 차이를 보고합니다. 신규 피험자·세션 검증은 별도로 필요합니다."]
    if any(spec["family"] in {"support_extra_trees", "support_soft_voting"} for spec in recipes):
        return ["1. 무변형 점수 경로의 완전 일치를 확인한 뒤, 개별 기준을 통과한 트리 수·점수 완화 구성만 추가 시드와 호환 결합으로 검증합니다.",
            "2. 두 전략 모두 실패하면 트리 수나 완화 강도를 계속 늘리지 않습니다. 지속적인 기록 단위 오류와 모델 시드 변동을 구분해 다음 가설의 근거를 검토합니다.",
            "3. 신규 피험자·세션과 실제 센서 배치를 포함한 평가를 준비합니다. 현재 반복 개발 자료의 점수는 전향적 성능을 입증하지 않습니다."]
    if any(spec["family"] in {"interaction_extra_trees", "interaction_soft_voting"} for spec in recipes):
        return ["1. 무변형 대조군의 완전 일치를 먼저 확인하고, 성공한 개별 특징 변경만 추가 시드 및 호환 결합 실험으로 진행합니다.",
            "2. 두 변경 모두 실패하면 같은 변환의 세부 설정을 결과에 맞춰 늘리지 않습니다. 기존 고차원 표현과 다른 사전 지정 저차원 상호작용 표현의 근거를 검토합니다.",
            "3. 신규 피험자·다른 날의 재착석과 센서 위치 메타데이터를 확보해 반복 사용한 개발 결과의 한계를 점검합니다."]
    if any(spec["family"] in {"record_prototype", "class_conditional_mixture", "local_soft_voting"} for spec in recipes):
        return ["1. 개발 기준을 통과한 후보가 있을 때만 추가 시드와 고정 평가를 진행하고, 호환되는 개별 성공 후보만 결합합니다.",
            "2. 두 지역 분포 접근이 모두 실패하면 거리·혼합 성분 수를 평가 결과에 맞춰 늘리지 않습니다. 훈련 전용 표현 압축이나 센서 물리적 배치의 근거를 먼저 검토합니다.",
            "3. 신규 피험자·다른 날의 재착석 데이터를 확보해 반복 사용한 개발 분할의 낙관성을 점검합니다. 현재 평가를 신규 모집단 성능으로 해석하지 않습니다."]
    if any(spec["family"] in {"record_gain_extra_trees", "gain_soft_voting"} for spec in recipes):
        return ["1. 증강이 개발 기준을 통과하면 추가 시드에서도 유지되는지 먼저 확인하고, 실제 센서 교정 강건성은 별도 실측으로 확인합니다.",
            "2. 증강이 모두 실패하면 강도를 결과에 맞춰 반복 조정하지 않습니다. 이미 실패한 표현·규제와 구분되는 훈련 전용 거리 또는 지역 구조 가설의 근거를 먼저 검토합니다.",
            "3. 독립된 신규 피험자·세션과 실제 착석 조건에서 검증할 자료를 준비합니다. 현재 개발 반복 탐색으로 새로운 모집단 성능을 대신하지 않습니다."]
    if any(spec["family"] in {"ordinal_extra_trees", "shrinkage_qda", "spatial_soft_voting"} for spec in recipes):
        return ["1. 개발·추가 시드 기준을 통과한 표현이 있다면 개별 효과와 결합 효과를 분리해 확인합니다. 실패한 표현을 점수 조정으로 재채택하지 않습니다.",
            "2. 센서 이득 증강은 원시 센서 단위와 허용 범위를 확인한 뒤, 훈련 기록 내에서만 일정한 작은 변화로 사전 등록합니다. 원본 중복 대조군으로 단순 표본 수 증가 효과를 구분합니다.",
            "3. 독립된 신규 피험자·재착석 세션을 확보해 반복 사용한 개발 점수의 낙관성을 점검합니다. 현재 코호트 실험과 새 코호트 결과를 섞지 않습니다."]
    if any(spec["family"].startswith("record_") for spec in recipes):
        return ["1. 채택된 Extra Trees의 Gini 분할을 엔트로피 분할과 비교합니다. 트리 수·깊이·특징·기록 정책은 유지합니다.",
            "2. 사후 비용-복잡도 가지치기의 소수 alpha를 사전 고정해 비교합니다. 기존 규제에 더해지는 과소적합 위험을 확인합니다.",
            "3. 분할 전 최소 불순도 감소 기준을 별도로 비교합니다. 각 접근의 성공이 확인되기 전에는 조합하지 않습니다."]
    if any(spec["family"] == "extra_trees" and any(key in spec["model_params"]
            for key in ("criterion", "ccp_alpha", "min_impurity_decrease")) for spec in recipes):
        return ["1. 센서 평균의 모든 쌍별 순서 비교 특징을 기존 압력 특징에 추가해 위치 간 접촉 순서 가설을 검증합니다. 검증 라벨로 비교 쌍을 선택하지 않습니다.",
            "2. 사전 지정한 저차원 센서 요약에서 강하게 정규화한 클래스별 공분산 결정 경계를 독립 후보로 비교합니다.",
            "3. 훈련 기록 단위로 일정한 센서 이득 변화를 주는 증강은 물리적 타당성과 작은 사전 고정 범위를 먼저 검토합니다. 원본 및 평가 입력을 수정하지 않습니다."]
    return ["1. 기록별 가중치 또는 기록 단위 재표집을 훈련 분할 안에서만 적용하고, 피험자 균형 가중치와 분리해 비교합니다.",
        "2. 개발에서 통과한 특징 표현이 있을 때만 소규모 규제를 탐색하고 추가 시드에서 효과가 유지되는지 확인합니다.",
        "3. SVM을 계속 검토한다면 피험자 분리 OOF 확률 보정을 훈련 데이터 안에서만 구성하고 분류 Macro F1도 함께 확인합니다."]


def _source_hashes_are_valid(out, protocol):
    for relative, expected in protocol["source_hashes"].items():
        if digest(out / "source_snapshot" / relative) != expected:
            raise ValueError(f"Frozen source changed: {relative}")


def render(out: Path, report: Path, checkpoint_name=None, checkpoint_version="1.0.0") -> str:
    # This guard precedes any report creation, including its parent directory.
    if not (out / "evaluation.json").is_file():
        raise ValueError("평가가 끝나지 않았습니다. evaluation.json 생성 후 보고서를 작성하세요.")
    p, selection, evaluation = (read(out / name) for name in ("protocol.json", "selection_lock.json", "evaluation.json"))
    protocol_hash, selection_hash = digest(out / "protocol.json"), digest(out / "selection_lock.json")
    if read(out / "protocol_lock.json")["sha256"] != protocol_hash:
        raise ValueError("Protocol lock changed")
    if selection["protocol_sha256"] != protocol_hash or read(out / "selection_seal.json")["sha256"] != selection_hash:
        raise ValueError("Selection seal changed")
    if read(out / "evaluation_seal.json")["sha256"] != digest(out / "evaluation.json"):
        raise ValueError("Evaluation seal changed")
    for name, key in (("dataset.json", "dataset_sha256"), ("data.npz", "cache_sha256")):
        if digest(out / name) != p[key]:
            raise ValueError(f"Frozen input changed: {name}")
    _source_hashes_are_valid(out, p)
    if p.get("identity_controls"):
        if not (out / "audits/identity_controls.json").is_file():
            raise ValueError("Preregistered identity control evidence is missing")
        from experiments.iterative_improvement.run import verify_identity_controls
        verify_identity_controls(out, p)
    dataset = read(out / "dataset.json")
    selected = selection.get("selected")
    winner = evaluation.get("improved") is True and evaluation.get("meaningful_gain") is True
    final_open = evaluation.get("holdout_evaluated") is True
    reproduction = read(out / "reproducibility.json") if (out / "reproducibility.json").is_file() else None
    if winner and (not selected or not reproduction or reproduction.get("reproduced") is not True):
        raise ValueError("개선 체크포인트의 후보·재현 검증 근거가 불완전합니다.")
    if final_open and evaluation.get("selection_sha256") != selection_hash:
        raise ValueError("Evaluation selection differs")
    incumbent = [verified_fit(out, out / "development/baseline" / part["id"], protocol_hash) for part in p["partitions"]]
    candidates = selection["candidates"]
    identity_controls = set(p.get("identity_controls", ()))
    candidate_rows = {row["id"]: [verified_fit(out, out / "development" / row["id"] / part["id"], protocol_hash)
        for part in p["partitions"]] for row in candidates}
    for row in candidates:
        f1 = [record["metrics"]["macro_f1"] for record in candidate_rows[row["id"]]]
        if f1 != row["f1"] or abs(mean(f1) - row["mean_f1"]) > 1e-12:
            raise ValueError(f"Selection metrics differ: {row['id']}")
    adoption = p.get("user_adoption")
    adopted = bool(adoption and selected and reproduction and reproduction.get("reproduced") is True and final_open)
    authorization = None
    if adoption:
        if digest(out / "user_authorization.json") != adoption["authorization_sha256"]:
            raise ValueError("User adoption authorization changed")
        authorization = read(out / "user_authorization.json")
    name = checkpoint_name or (f"{out.name}__{selected['id']}" if winner or adopted else "없음")
    status = "사용자 채택 연구 체크포인트" if adopted else "검증된 개선 후보" if winner else "신규 검증 체크포인트 없음"
    lines = [f"# 반복 모델 개선 실험: {out.name}", "",
        f"**{status}.** " + ("사용자의 명시적 요청에 따라 일반 정자세 기록을 복원한 Extra Trees를 다음 연구 기준으로 채택합니다. 원래 자동 채택 기준 충족 여부와 별도 결정입니다."
            if adopted else "사전에 정한 개발·고정 평가·재현 기준을 모두 통과했습니다."
            if winner else "이 반복에서 채택 기준을 모두 충족한 개선이 없어 현재 연구 기준 모델을 유지합니다."), "",
        "## 체크포인트 식별과 결정", "",
        f"- 체크포인트명: `{name}`", f"- 체크포인트 버전: `{checkpoint_version}`" if winner or adopted else "- 체크포인트 버전: 해당 없음",
        f"- 실행 ID: `{out.name}`", f"- 선택 후보: `{selected['id']}`" if selected else "- 선택 후보: 없음",
        f"- 실제 실행 시작: {p['started_at']}", f"- 실제 실행 종료: {evaluation['finished_at']}",
        "- 위 시각은 학습·평가 실행 시각이며 센서 취득 시각이 아닙니다. 원 측정 시각이 없는 기록의 취득일은 추정하지 않습니다.",
        "- 기존 기준 모델 파일과 활성 모델은 보존합니다. 연구 체크포인트 검증, 모델 추가에서의 등록, 런타임 승인·활성화는 별도 단계입니다.",
        "- 이 보고서는 등록·승인·활성화를 수행하지 않습니다. 모델·보고서·프로토콜·평가·재현 근거를 묶어 상태에 맞게 등록해야 합니다.", "",
        "## 목적과 가설", "",
        "동일한 피험자 분리 프로토콜에서 현재 연구 기준 모델보다 의미 있게 좋아지는 구성만 누적합니다. 모든 후보는 동일한 입력 창과 정답, 개발 분할 및 지표를 사용합니다.", ""]
    lines += table(["후보", "변경·검증 가설"], [(row["id"], hypothesis(row["pipeline"],
        identity_control=row["id"] in identity_controls)) for row in candidates])
    if authorization:
        lines += ["## 사용자 채택 결정과 기준 예외", "",
            f"요청 원문: **{authorization['instruction']}**", "",
            f"요청 기록 시각: `{authorization['authorized_at']}`. 기존 반복004의 개발 개선 {delta(authorization['observed_development_delta'])}가 +1.00%p에 근접했으므로 해당 구성만 채택하라는 결정입니다. "
            "원래 보고서·선택 봉인·평가 결과는 변경하지 않았습니다. 새 확인 실행에서는 이 후보에 한해 개발 진입 기준을 +0.90%p로 기록했고, 다음 자율 반복의 기본 기준 +1.00%p는 유지합니다.", "",
            f"- 사용자 채택 상태: `user_adopted_research`; 체크포인트 ID: `{authorization['checkpoint_id']}`.",
            f"- 원래 자동 채택 기준 전체 충족: `{evaluation.get('original_autonomous_criteria_met', False)}`.",
            f"- 이번 확인 실행의 개발·고정 평가·추가 시드 기준 전체 충족: `{winner}`.",
            "- 사용자 결정은 통계적 우월성의 증거가 아닙니다. 아래 고정 평가·불확실성·자세별 변화와 재현 결과를 그대로 남깁니다.",
            "- 기존 Extra Trees 비교 기준과 운영 활성 RBF-SVM을 보존합니다. 향후 실험은 이 사용자 채택 구성에서 시작하되 이전 기준과도 비교할 수 있습니다.",
            link(report, out / "user_authorization.json", "불변 사용자 요청 기록"), ""]
        if final_open:
            primary = evaluation["final"][0]
            interval = evaluation["paired_subject_interval"]
            lines += [f"**추가 고정 평가 결과:** 주 시드 Macro F1은 기존 {percentage(primary['baseline']['macro_f1'])} → "
                f"채택 구성 {percentage(primary['candidate']['macro_f1'])}로 {delta(primary['delta'])} 변했습니다. "
                f"피험자 bootstrap 차이의 기술적 95% 구간은 [{delta(interval['lower'])}, {delta(interval['upper'])}]입니다. "
                "이 결과를 개발 점수 상승과 함께 판단해야 하며, 채택을 고정 평가 성능 향상으로 표시하지 않습니다.", ""]
    lines += ["## 기준 모델과 이전 체크포인트 비교", "",
        f"기준 모델은 `{p['baseline']['family']}`이며 개발 Macro F1 평균은 **{percentage(mean(r['metrics']['macro_f1'] for r in incumbent))}**입니다. "
        "앱에서 활성화된 모델과 이번 연구 비교의 기준 모델을 구분합니다.", "",
        f"기준 선정 근거: `{p.get('incumbent_basis', '프로토콜 참조')}`.", ""]
    if p.get("baseline_checkpoint"):
        checkpoint_ref = p["baseline_checkpoint"]
        lines += ["이번 반복의 시작 구성은 " + link(report, ROOT / checkpoint_ref["path"], "등록된 연구 기준 체크포인트")
            + f"이며 명세 SHA-256은 `{checkpoint_ref['sha256']}`입니다. 이전 사용자 채택은 자동 검증 통과와 구분하며, 이번 새 후보에는 기본 +1.00%p 개발 기준을 다시 적용합니다.", ""]
    lines += table(["개발 분할", "학습 피험자", "검증 피험자", "Macro F1", "Accuracy", "최저 클래스 F1"], [
        (part["id"], len(part["train"]), len(part["validation"]), percentage(row["metrics"]["macro_f1"]),
         percentage(row["metrics"]["accuracy"]), percentage(row["metrics"]["worst_class_f1"]))
        for part, row in zip(p["partitions"], incumbent)])
    origin = (out / "dataset.json").resolve().parent
    prior = origin / "selection_lock.json"
    if prior.is_file() and prior != out / "selection_lock.json":
        old = read(prior)
        same = [(key, row) for key, row in old.get("selected", {}).items()
            if row.get("family") == p["baseline"]["family"] and row.get("params") == p["baseline"]["model_params"]]
        if same:
            key, row = same[0]
            lines += [f"이전 실험 `{key}`와 동일한 모델 파라미터를 사용합니다. 이전 개발 평균 Macro F1은 "
                f"{percentage(row['mean_macro_f1'])}이며 이번 기준 재평가와의 차이는 "
                f"{delta(mean(r['metrics']['macro_f1'] for r in incumbent) - row['mean_macro_f1'])}입니다. "
                f"{link(report, prior, '이전 후보 선택 잠금')}에 원 설정이 있습니다.", ""]
    lines += [("이번 구성은 사용자 명시 채택으로 연구 기준을 변경하며 자동 개선 통과와 구분합니다."
        if adopted else "개선이 없으면 신규 체크포인트를 만들지 않습니다. 후보의 수치가 일부 높더라도 모든 채택 기준을 충족하기 전에는 기준을 바꾸지 않습니다."), "",
        "### 기준 모델의 정확한 구성", ""] + code(p["baseline"])
    lines += ["## 입력과 학습·평가 설정", ""]
    lines += table(["항목", "설정"], [
        ("코호트", f"{p['cohort']['subjects']}명 / {p['cohort']['records']}개 기록 / {p['cohort']['windows']}개 창"),
        ("입력", "공통 3초 / 12 Hz / 36×16 센서값; 통계 모델은 기존 596개 특징, 시계열 모델은 동일 창의 36×16 배열"),
        ("분할", "피험자와 원 기록이 겹치지 않는 고정 개발 분할; 고정 평가 피험자는 개발에서 제외"),
        ("주 지표", "7개 클래스 Macro F1; 부 지표 Accuracy, 클래스별 F1, 피험자별 결과"),
        ("후보 선택", p["selection"]), ("기본 캐시·평가의 등록 기록 처리", p["registration_policy"] +
            ("; TRAIN에 한하여 아래 사전 지정 일반 정자세 복원 정책 적용. 실제 등록·교정 기록은 제외." if p.get("supplemental_training") else "")),
        ("시드", p["seeds"]), ("전처리", "특징·입력 표현은 사전 정의; scaler·whitening·시계열 임계값은 매 학습 분할에서만 적합"),
        ("최종 재학습", "개발 학습·검증 피험자의 합집합; 고정 평가 데이터로 모델 선택·조기 종료하지 않음"),
        ("결합", p["combination"]),
    ])
    if p.get("supplemental_training"):
        extra = p["supplemental_training"]
        if digest(out / "supplemental_manifest.json") != extra["manifest_sha256"] or digest(out / "supplemental_training.npz") != extra["cache_sha256"]:
            raise ValueError("Supplemental training evidence changed")
        lines += ["### 사전 등록한 일반 정자세 학습 기록 복원", "",
            f"추가 캐시는 기존 32명에 속한 일반 정자세 {extra['records']}개 기록·{extra['windows']}개 창입니다. "
            "해당 정책을 명시한 후보에만 각 분할의 학습 피험자 기록을 추가합니다. 검증·고정 평가 피험자의 추가 기록은 사용하지 않으며 기존 평가 창은 동일합니다. "
            "현재 메타데이터와 학습 적격성으로 일반 기록임을 확인했고, 실제 등록·교정으로 표시된 파일은 허용하지 않습니다. "
            "대부분의 과거 CSV에 취득 목적의 별도 sidecar가 없어 물리적 측정 목적을 독립적으로 확증한 것은 아닙니다.", "",
            "클래스 빈도 기반 가중치를 사용하는 구성은 추가 후 학습 행에서 가중치를 계산합니다. 다른 구성은 아래 후보별 동일 사전확률·역거리 점수 등 명시한 학습 규칙을 따릅니다. 개인 정자세 기준 벡터를 입력으로 사용하지 않습니다. "
            "모델 추가에서는 사용자가 선택한 적격 기록으로 학습하는 기존 동작을 사용하고, 연구에서의 기록 복원 정책과 출처를 파이프라인 메타데이터에 보존합니다.", "",
            link(report, out / "supplemental_manifest.json", "추가 기록의 경로·해시·적격성·정책 명세"), ""]
        lines += table(["후보", "개발 분할", "추가 기록", "추가 창", "정자세 창 변화"], [
            (name, part["id"], row["supplemental_training_audit"]["added_records"], row["supplemental_training_audit"]["added_windows"],
             str(row["supplemental_training_audit"]["class_counts_before"][0]) + " → " + str(row["supplemental_training_audit"]["class_counts_after"][0]))
            for name, rows in candidate_rows.items() for part, row in zip(p["partitions"], rows) if row.get("supplemental_training_audit")])
    lines += ["각 분할의 피험자와 원본 파일 SHA-256은 " + link(report, out / "dataset.json", "고정 데이터 명세") + "에 기록되어 있습니다.", "",
        "### 실행 전 고정한 채택 기준", "",
        f"1. 개발 평균 Macro F1이 기준 대비 {delta(p['criteria']['minimum_development_delta'])} 이상이며 "
        f"{p['criteria']['minimum_positive_splits']}개 이상의 개발 분할에서 상승해야 합니다.",
        f"2. 개발 분할 평균 클래스별 F1과 주 시드 고정 평가 클래스별 F1은 각각 최대 {percentage(p['criteria']['maximum_per_class_drop'])}p보다 크게 떨어지면 안 됩니다.",
        "3. 선택 후보를 봉인한 뒤 추가 시드로 개발 개선을 확인합니다. 각 추가 시드의 개발 평균 차이가 양수여야 고정 평가를 엽니다.",
        f"4. 주 시드 고정 평가 Macro F1이 {delta(p['criteria']['minimum_holdout_delta'])} 이상, 모든 고정 평가 확인 시드의 차이가 양수여야 합니다.",
        f"5. 피험자 단위 paired bootstrap {p['criteria']['bootstrap_replicates']:,}회 차이의 95% 구간 하한이 0을 넘어야 합니다.",
        f"6. 같은 시드 재학습과 저장 모델 재로드의 최대 확률 차이가 각각 {p['criteria']['refit_probability_atol']:g} 이하여야 합니다.", ""]
    lines += code(p["criteria"])
    if p.get("identity_controls"):
        lines += ["### 무변형 처리 경로 대조군", "",
            "후보 선택 전에 사전 등록한 무변형 대조군과 기준의 모든 개발 예측 배열을 비교했습니다. 정답·행·기록·피험자·예측·확률의 "
            "형상·자료형·바이트와 지표가 완전히 같아야 통과합니다. 불일치하면 성능 선택을 중단합니다. "
            "사전에 정의한 무변형 처리 경로가 기존 기준과 같은 동작을 하는지 확인하는 구현 검증이며 새 성능 개선 증거는 아닙니다.", "",
            link(report, out / "audits/identity_controls.json", "분할별 무변형 대조군 예측 해시 감사"), ""]
    if any(row["pipeline"]["family"] in {"record_balanced_extra_trees", "record_bagged_extra_trees", "record_soft_voting"} for row in candidates):
        audit_path = out / "audits/training_record_structure.json"
        audit = read(audit_path)
        if (audit["protocol_sha256"] != protocol_hash or audit["cache_sha256"] != p["cache_sha256"]
                or audit["supplemental_cache_sha256"] != p["supplemental_training"]["cache_sha256"]):
            raise ValueError("Training record structure audit differs from frozen inputs")
        lines += ["### 훈련 기록 구성과 가설의 영향 규모", "",
            "아래는 모델 적합이나 평가 예측 없이 훈련 행에서만 계산한 값입니다. 사람별 총 기여를 같게 만들지 않으며, "
            "기록 균형 가중치의 변화 폭과 별도로 기록 단위 재표집을 비교합니다.", ""]
        lines += table(["개발 분할", "학습 기록", "학습 창", "기록당 창 수 분포", "기존 클래스 가중치 대비 비율", "평균 절대 가중치 변화"], [
            (row["partition"], row["training_records"], row["training_windows"], row["record_window_count_distribution"],
             f"{row['record_weight_over_baseline_class_weight']['min']:.4f}–{row['record_weight_over_baseline_class_weight']['max']:.4f}",
             percentage(row["record_weight_over_baseline_class_weight"]["mean_absolute_change"])) for row in audit["rows"]])
        lines += [link(report, audit_path, "훈련 기록 구조 감사와 입력 해시")
            + f" · 감사 파일 SHA-256: `{digest(audit_path)}`", ""]
    lines += ["## 후보별 변경과 개발 결과", "",
        "아래의 ‘개발 통과’는 최종 체크포인트 채택과 다릅니다. 같은 고정 기준으로 비교한 결과입니다.", ""]
    lines += table(["후보", *[part["id"] for part in p["partitions"]], "평균 F1", "기준 대비", "개발 통과"], [
        (row["id"], *[percentage(v) for v in row["f1"]], percentage(row["mean_f1"]), delta(row["mean_delta"]),
         "대조군 · 선택 제외" if row["id"] in identity_controls else "예" if row["qualified"] else "아니오") for row in candidates])
    lines += table(["후보", "도입한 변경", "모델 파라미터"], [
        (row["id"], recipe_description(row["pipeline"]), json.dumps(row["pipeline"].get("model_params", {}), ensure_ascii=False, sort_keys=True))
        for row in candidates])
    lines += ["### 성공·실패와 해석", ""]
    if identity_controls:
        changes = [row for row in candidates if row["id"] not in identity_controls]
        passed = sum(row["qualified"] is True for row in changes)
        lines += [f"무변형 대조군 {len(identity_controls)}개는 경로 검증을 통과했습니다. 성능 비교 후보 {len(changes)}개 중 "
            f"개발 기준 통과는 {passed}개, 미달은 {len(changes) - passed}개입니다.", ""]
    for row in candidates:
        if row["id"] in identity_controls:
            lines += [f"- `{row['id']}`: 무변형 경로 검증 통과 · 성능 선택 대상 아님. "
                f"모든 개발 분할 {len(row['f1'])}개에서 기준과 예측 배열·확률·지표가 완전히 일치했습니다."]
            continue
        positive = sum(v > 0 for v in row["deltas"])
        conclusion = ("개발 기준 통과; 최종 평가·재현 결과를 추가 확인해야 하는 후보" if row["qualified"]
            else "개발 기준 미달로 채택하지 않음")
        lines += [f"- `{row['id']}`: {conclusion}. 평균 변화 {delta(row['mean_delta'])}, 상승 분할 {positive}/{len(row['deltas'])}. "
            f"평균 클래스별 변화의 최솟값은 {delta(min(row['mean_class_deltas']))}입니다."]
    lines += ["", "관측된 상승·하락은 아래 후보별 가설과 함께 해석합니다. 바뀐 표현이나 학습 목적함수가 유용한 구분을 강화하거나 약화했을 가능성은 있지만, "
        "이 수치만으로 원인을 확정하지 않으며 센서 이상·라벨 오류·신체 차이를 입증하지 않습니다.", ""]
    if any(row["pipeline"]["family"] in {"coupled_extra_trees", "subject_residual_extra_trees", "structured_soft_voting"} for row in candidates):
        lines += ["OVO는 사전순 (0,1)…(5,6)의21개 쌍마다 해당 두 자세의 TRAIN 행만 원 순서로 사용합니다. "
            "공유 scaler는 전체 TRAIN에 한 번 적합하고, 이진 라벨은 i→0/j→1, seed=(s+pair_index)%2**32입니다. "
            "각 head는32개 트리·깊이8·최소 잎8·특징비율0.5·bootstrap=False·balanced를 사용합니다. "
            "총672개 트리이며 각 행이6개 head에 들어가192N tree×row 노출입니다. 기준200N과 용량·시간이 같다는 뜻은 아닙니다.", "",
            "r_ij=P(i|i,j)를 [1e-7,1-1e-7]로 제한하고 r_ji=1-r_ij로 둡니다. "
            "Q_ii=Σ(j≠i)r_ji², Q_ij=-r_ji*r_ij; min pᵀQp, Σp=1의 equality-KKT 8×8 시스템을 float64로 직접 풉니다. "
            "KKT condition≤1e8, scaled backward error·정규화 stationarity≤1e-12, 엄격히 양수 p와 합 오차≤1e-12를 요구합니다. "
            "한 번 합 정규화하고 모든21개 입력이 정확히0.5일 때만 정확한1/7로 동률을 고정합니다. 실패는 오류이며 대체 solver·음수 자르기·임의 softmax를 사용하지 않습니다. "
            "Q가 특이해도 KKT는 가역일 수 있으므로 Q condition으로 거절하지 않습니다. 이는 libsvm 반복 구현과 같은 목적함수이며 출력 바이트 일치를 주장하지 않습니다. "
            "근거: [Wu·Lin·Weng(2004), §4.1–4.2와 Appendix A/B](https://www.jmlr.org/papers/volume5/wu04a/wu04a.pdf). "
            "쌍별 균형은 일관성 가정 아래 균등 클래스 사전확률을 목표로 하지만 calibration 보장은 아닙니다. 앱의 confidence threshold는 바꾸지 않습니다.", "",
            "사람 방향 제거는 표준화된 각 TRAIN 기록의 창 평균, 사람×자세 안 기록 균등 평균, 사람 안7자세 균등 평균으로 사람 중심점을 만듭니다. "
            "산술은 각 특징의 값을 오름차순 정렬한 float64 평균이며 사람 평균으로 중심화한B의 full SVD 상위3방향V를 사용합니다. "
            "각 방향은 절댓값 최대 loading을 양수로 고정합니다. tol=eps64×max(사람 수,596)×s0, s2>tol 및 s2-s3>tol이 필수입니다. "
            "부족한 rank나3/4경계 동률은 실패로 기록하고 차원·solver를 바꾸지 않습니다.", "",
            "변환은 z-(zVᵀ)V이며 저장된 사람 평균을 추론 입력에서 빼지 않습니다. C-order float64·고정 순서 einsum(optimize=False), 재표준화 없음, native ET의 최종float32 변환을 사용합니다. "
            "평가 사람 ID·기록 평균·교정 입력은 필요하지 않습니다. 최소4명 및 사람마다7자세 TRAIN 범위를 검증합니다. "
            "제거된 사람 중심점 에너지·자세 중심점 대비 에너지는 기술 통계이며 차원 선택에 사용하지 않습니다. "
            "005의 열별 사람 F 페널티 선택이나017의 전체 창 PCA64 유지와 다르지만, 이 방향들이 순수 방해 요인이라는 보장은 없습니다. "
            "정자세·앞·뒤 기록 거리 진단에서도 같은 사람이면 명확히 구분된다는 전제는 지지되지 않았습니다.", ""]
    if any(row["pipeline"]["family"] in {"hardness_extra_trees", "hardness_random_forest", "hardness_soft_voting"} for row in candidates):
        lines += ["비복원 Random Forest는 bootstrap=False로 모든 TRAIN 창을 사용하고, 200개 트리·깊이8·최소 잎8·특징 비율0.5를 유지합니다. "
            "기준 Extra Trees와 원 특징596개·클래스 균형·표준화·일반 정자세 복원 정책이 같습니다. 임곗값 탐색을 포함한 backend의 분할 방식 비교이며, "
            "과거 bootstrap=True RF 결과와 직접적인 한 변수 대조라고 해석하지 않습니다. 탐색 비용과 트리 간 상관 증가가 가능한 한계입니다.", "",
            "기록 난이도 구성은 각 외부 TRAIN의 사람들만 GroupKFold4(shuffle=False)로 나누고, inner TRAIN만으로 scaler와 기준 Extra Trees를 적합합니다. "
            "inner seed=(외부 seed+1000+fold_index)%2**32이며 fold_index는0부터3입니다. 자기 사람을 학습하지 않은 OOF argmax 오류의 기록별 평균을 e_r로 둡니다. "
            "h_r=1+0.5e_r, sample_weight_i=h_record(i)/mean_{j:y_j=y_i}(h_record(j))이고, 분모는 기록 평균이 아닌 클래스 안의 창 평균입니다. "
            "native class_weight='balanced'와 곱한 클래스별 총 질량은 N/7로 유지됩니다. 최종 scaler는 전체 외부 TRAIN 창을 비가중 적합합니다.", "",
            "이 과정은 OOF 오류를 후처리 편향으로 사용한006이나 기록별 균등화009와 다릅니다. 학습 순서를 다시 정하거나 원본을 제외·수정하지 않습니다. "
            "어려운 기록을 강조하면 유효한 변이를 배울 수도 있지만 라벨 불확실성이나 사람 특이 패턴을 강화할 수도 있습니다. "
            "OOF 오류율은 외부 개발 성능 개선 증거가 아니고, class 총 질량 보존이 기록·사람별 균등화를 뜻하지 않습니다. "
            "coefficient=0 대조군은 OOF와 sample_weight 인자 자체를 생략하여 기존 ET를 정확히 재현합니다.", ""]
    if any(row["pipeline"]["family"] in {"projection_extra_trees", "projection_soft_voting"} for row in candidates):
        lines += ["PCA64는 TRAIN StandardScaler 뒤의 float64 특징에 full SVD를 적합하고 whitening 없이64개 축만 유지합니다. "
            "596개 특징을64개로 줄이면 max_features=0.5의 분할 후보 수도298개에서32개로 바뀝니다. "
            "분산은 클래스 구분력과 다르며 낮은 분산의 유용한 방향을 버릴 수 있습니다. 기존 정적 열 선택이나 고정 센서 Hadamard 변환과 구분합니다.", "",
            "LDA 추가 구성은 동일 TRAIN의 표준화 특징에서 상수 열을 제외하고 eigen solver·자동 공분산 수축·경험적 클래스 사전확률로6개 판별 축을 적합합니다. "
            "각 축의 절댓값이 가장 큰 loading을 양수로 고정한 뒤 원 표준화596개 좌표와 함께 사용합니다. 총602개·분할 후보301개로 달라집니다. "
            "005의6개 축 대체 표현과 다르지만, 그 실패가 원래 축의 제거 때문이었다는 근거는 없습니다. 새 미검증 표현 가설이며 개별 성공 없는 모델 결합으로 간주하지 않습니다.", "",
            "학습 투영 입력과 산술은 float64이며 저장된 축에 고정 순서 einsum(optimize=False)을 적용합니다. PCA는 저장 평균을 빼고, eigen LDA는 추가 중심화 없이 투영합니다. "
            "투영 뒤 재표준화하지 않으며 native ExtraTrees가 최종 좌표를 float32로 변환합니다. 무변형 대조군은 원 표준화 특징의 자료형·값을 그대로 전달합니다. "
            "PCA64를 지지하지 못하는 수치적 순위나 특이 LDA 공분산은 오류로 처리하고 자동 차원 축소·다른 solver로 우회하지 않습니다. "
            "반복 고유값의 기저는 수치 라이브러리에 따라 달라질 수 있어 환경·학습 상태 저장과 추가 재현 검증이 필요합니다.", ""]
    if any(row["pipeline"]["family"] in {"tail_extra_trees", "ovr_extra_trees", "alternatives_soft_voting"} for row in candidates):
        lines += ["특징 범위 제한은 전체 주변 분포를 정규분포로 다시 매핑한 이전 분위수 변환과 다릅니다. "
            "TRAIN에서 추정한 양끝 경계 밖만 제한하고 중앙 구간 값은 유지합니다. 경계 계산은 float64 linear quantile이며 "
            "제한 결과를 원 입력 float32로 바꾼 다음 StandardScaler를 적합합니다. 같은 경계를 개별 평가·추론 창에 적용하며 "
            "평가 모집단이나 현재 추론 배치에서 경계를 다시 계산하지 않습니다. 원본 측정·캐시를 수정하거나 행을 삭제하지 않습니다.", "",
            "일부 TRAIN 특징의 중앙 구간이 전체 범위보다 좁았다는 사실은 노드별 임곗값 낭비나 성능 개선을 입증하지 않습니다. "
            "값이 큰 기록이 실제 자세·사람 차이를 표현할 수 있으므로 제한에 따른 정보 손실도 가능한 설명입니다. "
            "무변형 대조군은 min/max 제한조차 하지 않고 원 특징을 그대로 전달합니다.", "",
            "OVR은 7개 head 각각이 동일 TRAIN 전체 행의 한 자세 대 나머지를 학습합니다. 이전 세 자세 계층 분류와 달리 gate와 조건부 확률 곱이 없습니다. "
            "각 head의 balanced 가중치는 양성과 음성의 총 질량을 같게 하며 음성 내부 자세 구성은 관측 빈도를 유지합니다. "
            "양성 점수를 합으로 나누는 것은 argmax를 바꾸지 않고 확률 보정을 의미하지 않습니다. 모든 양성 점수가 정확히 0인 경우만 고정 1/7로 처리합니다.", "",
            "head당32개·200개는 총224개·1400개 트리입니다. 기준의200개와 용량이 정확히 같지 않으므로 성과가 있더라도 분할 목적함수만의 효과로 해석하지 않습니다. "
            "개선이 없다면 클래스 간 점수 비교 편향이나 원 입력의 사람 간 차이가 남았을 수 있습니다. 비용은 실제 측정해야 하며, "
            "한 시드의 점수로 구조 전체의 우월성이나 열등성을 확정하지 않습니다.", ""]
    if any(row["pipeline"]["family"] in {"support_extra_trees", "support_soft_voting"} for row in candidates):
        lines += ["트리 수 증가는 같은 Extra Trees 분포에서 더 많은 무작위 트리를 평균하는 비교입니다. 시드별 변동을 줄일 수 있지만 "
            "편향이나 사람 간 압력 분포 차이를 직접 해결하지 않습니다. 이전 004의 400개 구성은 깊이·잎 크기·학습 기록 정책도 달랐으므로 트리 수만 바꾼 대조가 아니었습니다.", "",
            "잎 완화에서 p는 이미 클래스 가중치가 반영된 정규화 확률이고, n은 비가중 훈련 창 수입니다. p를 n으로 다시 나누지 않습니다. "
            "q = r×p + (1-r)/7, r=n/(n+alpha)이므로 각 트리의 클래스 순위는 유지하지만, 입력마다 도달한 잎의 크기가 달라 트리 간 영향이 바뀝니다. "
            "argmax에서 동일 클래스 항은 상쇄되므로 표본 수에 따른 트리 점수 가중으로도 해석할 수 있습니다. "
            "alpha=8일 때 n=8의 점수는 절반 유지되며, alpha=32이면 20% 유지됩니다. 유용한 작은 영역·드문 자세에 대한 점수도 약해질 수 있습니다.", "",
            "겹치는 창의 수는 독립 표본 수가 아니며 p×n은 관측 클래스 빈도가 아닙니다. 따라서 이 처리는 고정된 점수 규제이며 "
            "Dirichlet 사후확률이나 검증된 신뢰도로 표시하지 않습니다. alpha=0은 산술적으로 재계산하지 않고 기존 forest.predict_proba를 직접 호출합니다.", ""]
    if any(row["pipeline"]["family"] in {"interaction_extra_trees", "interaction_soft_voting"} for row in candidates):
        lines += ["고정 Hadamard 변환은 훈련 표준화 뒤에서 16차원 센서 통계 블록 32개에 각각 적용합니다. 절대·상대 특징 각각 256개를 변환하고 집계 특징 84개는 그대로 둡니다. "
            "실수 연산에서는 직교·가역이고 거리 보존이지만 구현은 float64 곱셈 후 float32로 한 번 반올림합니다. "
            "개선은 좌표축에 의존하는 트리 분할에 유용한 선형 조합이 생겼다는 가설과 양립하고, 하락은 원래 개별 센서 경계가 더 유용했다는 설명과 양립합니다. "
            "새 정보나 물리적 센서 배치 불변성이 생긴다는 뜻은 아닙니다.", "",
            "시간 상관은 36개 시점으로 이루어진 각 창 내부에서 센서별 평균을 뺀 신호를 사용합니다. 상수 센서가 포함된 쌍은 0이며, "
            "모든 120쌍을 고정 순서로 기존 596개 특징에 추가합니다. 짧고 보간된 창의 상관은 불안정하거나 공통 잡음을 표현할 수 있습니다. "
            "716차원에서 max_features=0.5의 명목상 분할 후보 수는 358개로 기준의 298개와 다릅니다. "
            "효과가 있더라도 상관 정보와 특징 경쟁 변화의 기여를 이 비교만으로 분리할 수 없습니다.", ""]
    if any(row["pipeline"]["family"] in {"record_prototype", "class_conditional_mixture", "local_soft_voting"} for row in candidates):
        lines += ["기록 대표점은 훈련 기록의 평균으로 창 내부 변동을 줄이지만, 평가 입력은 단일 창이므로 훈련 대표점과의 분포 차이가 남습니다. "
            "596차원의 상관된 특징에서 거리 집중이나 중복 특징의 기여가 생길 수 있고, 기록이 많은 사람의 영향도 완전히 제거하지 않습니다. "
            "역거리 정규화 값은 비교 점수이며 실제 정답 확률로 보정된 값이 아닙니다.", "",
            "클래스별 혼합분포는 복수 압력 양상을 표현할 수 있으나 대각 공분산 가정으로 특징 상관을 무시합니다. "
            "성분 수 1과 2의 차이는 이 설정에서 다중 양상의 효과를 비교하며, Extra Trees와의 차이는 분류기 전체의 변경을 포함합니다. "
            "성분이 사람 차이에 맞춰지거나 밀도가 과도하게 확신할 수 있습니다. 수렴 실패는 점수로 대체하지 않고 오류로 처리합니다. "
            "기록 대표점·표준화·밀도는 훈련 행에만 적합하고, 평가 창별 독립 추론과 같은 argmax 평가를 유지합니다.", ""]
    if any(row["pipeline"]["family"] in {"cnn1d", "minirocket", "sequence_soft_voting"} for row in candidates):
        lines += ["시계열 모델 비교에서는 시간 구조가 정적 자세의 사람 간 차이를 줄일 만큼 유용한지가 가설입니다. "
            "성능이 낮으면 36개 시점의 짧은 정적 기록에서 시간 변화보다 공간 압력 분포가 더 유용했거나, "
            "현재 학습량·규제·네트워크 구조가 맞지 않았을 가능성이 있습니다. 이 실험만으로 시계열 방식 전체가 열등하다고 결론내리지 않습니다. "
            "후보와 기준의 학습 피험자·복원 정책·평가 행은 같으며, 외부 검증으로 CNN epoch를 선택하지 않았습니다.", ""]
    if any(row["pipeline"]["family"] in {"record_balanced_extra_trees", "record_bagged_extra_trees", "record_soft_voting"} for row in candidates):
        lines += ["기록 기반 후보는 기존 특징과 학습 피험자를 유지하고 가중치 또는 훈련 기록 추출 단위만 바꿉니다. "
            "가중치 개선이 작다면 기록마다 창이 6–7개로 이미 비슷하다는 점과 양립합니다. 기록 재표집이 나빠지면 각 묶음의 유효한 기록 다양성 감소나 "
            "기준 Extra Trees 자체의 무작위화로 이미 충분히 분산을 줄였을 가능성이 있습니다. 성능 차이만으로 원인을 확정하지 않습니다. "
            "각 적합의 training_details에 실제 훈련 기록·가중치 또는 재표집 기록과 시드가 직렬화되어 있습니다.", ""]
    if any(row["pipeline"]["family"] in {"record_gain_extra_trees", "gain_soft_voting"} for row in candidates):
        lines += ["공통 배율과 센서별 배율은 각각 전체 수치 크기와 공간 분포를 교란하는 가설입니다. 개발 개선은 제한된 입력 변동에 대한 규제 효과와 양립합니다. "
            "하락한다면 유용한 절대 크기 또는 센서 간 차이를 변형했거나, 원 기록과 변형 기록을 동시에 보존하지 않는 단일 실현의 변동이 영향을 주었을 수 있습니다. "
            "사람·기록·창 수·클래스 가중치·트리 설정을 유지하지만 자세 라벨 보존의 물리적 타당성은 이 점수로 증명되지 않습니다. "
            "학습 창만 교체하고 평가 시에는 증강하지 않습니다. 기록별 배율과 시드 생성 규칙은 모델의 training_details에 저장합니다.", "",
            "현재 문서는 센서값의 범용 상한이나 교정 이득 허용 범위를 확정하지 않으므로 clipping·음수 제거·offset·채널 교환을 추가하지 않았습니다. "
            "양의 작은 배율만 사용하고 비유한 수치는 거부합니다. 실제 센서 강건성 및 신규 코호트 검증은 별도입니다.", ""]
    if any(row["pipeline"]["family"] in {"ordinal_extra_trees", "shrinkage_qda", "spatial_soft_voting"} for row in candidates):
        lines += ["순서 비교 특징의 이점은 공간 관계를 직접 표현한다는 가설과 양립합니다. 하락한다면 이산화로 크기 정보가 충분히 표현되지 않거나, "
            "원래 특징과 중복된 후보가 늘어 유용한 분할의 빈도가 달라졌을 수 있습니다. 원래 596개 특징은 제거하지 않았으므로 전체 정보 삭제로 해석하지 않습니다. "
            "평균과 중앙값을 함께 추가한 차이에는 차원 수 변화도 포함됩니다.", "",
            "QDA가 낮다면 고정 32개 평균에서 사라진 변동·공간 집계 정보, 클래스별 정규분포 가정, "
            "강한 공분산 수축에 따른 편향이 가능한 설명입니다. 이 실험은 특징 축소와 분류기 변경을 함께 비교하므로 각각의 원인 기여를 분리해 증명하지 않습니다. "
            "SVD에서 클래스별 공분산 고유값은 (1-reg_param)×추정 고유값 + reg_param으로 바뀝니다. "
            "모든 변환·평균·공분산은 해당 훈련 데이터에서만 적합합니다.", ""]
    if any(row["pipeline"]["family"] == "extra_trees" and any(key in row["pipeline"]["model_params"]
            for key in ("criterion", "ccp_alpha", "min_impurity_decrease")) for row in candidates):
        lines += ["분할 목적함수 변경이 일부 분할에만 유리하다면 참가자 구성에 따라 이점이 달라진다는 결과입니다. "
            "가지치기나 최소 불순도 감소 기준을 강화할수록 점수가 낮아지는 경우, 이미 깊이·잎 크기를 제한한 기준에 규제를 더해 "
            "유용한 자세 차이도 제거했을 가능성과 양립합니다. 정확한 원인을 증명한 것은 아니며, 일부 분할의 작은 상승을 전체 개선으로 해석하지 않습니다.", ""]
    combination = out / "combination_lock.json"
    if combination.is_file():
        combo = read(combination)
        lines += [f"결합 구성은 `{', '.join(combo['members'])}`를 후보 확정 후 같은 방식으로 다시 학습한 것입니다. "
            "최선의 단일 구성 대비로도 같은 개발 개선 기준을 요구했습니다. " + link(report, combination, "결합 잠금"), ""]
    else:
        lines += ["개별 개발 기준을 통과한 호환 후보가 2개 미만이어서 결합 실험을 진행하지 않았습니다.", ""]
    lines += ["## 추가 시드·고정 평가·재현 검증", ""]
    robust = evaluation.get("development_seed_robustness", evaluation.get("robustness", []))
    if robust:
        lines += table(["개발 확인 시드", "분할별 기준 대비 F1", "평균 변화"], [
            (row["seed"], ", ".join(delta(v) for v in row["deltas"]), delta(row["mean_delta"])) for row in robust])
    if not final_open:
        lines += ["이번 반복에서 고정 평가를 열지 않았습니다. 따라서 이번 후보의 고정 평가 우월성·재학습 재현 성공·신규 사용자 성능을 주장하지 않습니다.",
            f"평가 종료 기록: `{evaluation.get('reason', '평가 JSON 참조')}`.", ""]
    else:
        lines += table(["시드", "기준 F1", "후보 F1", "차이", "기준 Accuracy", "후보 Accuracy"], [
            (row["seed"], percentage(row["baseline"]["macro_f1"]), percentage(row["candidate"]["macro_f1"]), delta(row["delta"]),
             percentage(row["baseline"]["accuracy"]), percentage(row["candidate"]["accuracy"])) for row in evaluation["final"]])
        interval = evaluation["paired_subject_interval"]
        lines += [f"피험자 paired bootstrap 95% 구간: **[{delta(interval['lower'])}, {delta(interval['upper'])}]** "
            f"({interval['replicates']:,}회). 이미 관찰한 고정 피험자 및 선택 절차에 조건부인 기술적 구간입니다. "
            "반복 탐색·다중 비교를 보정한 독립 확증 검정이나 미래 모집단의 신뢰구간이 아닙니다.", ""]
        first = evaluation["final"][0]
        lines += ["### 주 시드의 클래스별 결과", ""]
        lines += table(["자세", "기준 F1", "후보 F1", "차이", "후보 Precision", "후보 Recall", "창 수"], [
            (label, percentage(first["baseline"]["per_class_f1"][label]), percentage(values["f1"]),
             delta(values["f1"] - first["baseline"]["per_class_f1"][label]), percentage(values["precision"]),
             percentage(values["recall"]), values["support"]) for label, values in first["candidate"]["per_class"].items()])
        final_dir = out / "final" / str(first["seed"])
        base = verified_fit(out, final_dir / "baseline", protocol_hash)
        candidate = verified_fit(out, final_dir / selected["id"], protocol_hash)
        lines += ["### 주 시드의 피험자별 결과", ""]
        lines += table(["피험자", "기준 F1", "후보 F1", "차이", "후보 Accuracy"], [
            (subject, percentage(base["per_subject"][subject]["macro_f1"]), percentage(values["macro_f1"]),
             delta(values["macro_f1"] - base["per_subject"][subject]["macro_f1"]), percentage(values["accuracy"]))
            for subject, values in candidate["per_subject"].items()])
        lines += ["### 주 시드 혼동행렬", "",
            "행은 실제 클래스, 열은 예측 클래스입니다. 클래스 순서는 데이터 명세의 class_names와 같습니다.", ""]
        lines += code({"class_names": dataset["class_names"], "baseline": first["baseline"]["confusion_matrix"],
            "candidate": first["candidate"]["confusion_matrix"]})
        lines += [f"일괄 추론 처리시간/창: 기준 {base['batch_inference_ms_per_window']:.4f} ms, "
            f"후보 {candidate['batch_inference_ms_per_window']:.4f} ms. 이는 한 번의 일괄 예측 측정이며 실제 센서의 3초 대기시간이나 GUI 응답시간 검증이 아닙니다.", ""]
    if reproduction:
        lines += table(["재현 항목", "결과"], [("같은 시드 재학습 최대 확률 오차", reproduction["max_probability_error"]),
            ("직렬화 모델 재로드 최대 확률 오차", reproduction["reload_max_error"]),
            ("재현 기준 통과", reproduction["reproduced"])])
    else:
        lines += ["최종 후보가 확정되지 않았거나 추가 시드 검증에서 중단되어 최종 재학습·재로드 재현 파일은 생성되지 않았습니다.", ""]
    lines += ["## 제한·위험과 다음 실험", "",
        "- 전체 코호트와 역사적 고정 평가 피험자는 프로젝트에서 이미 관찰한 자료입니다. 새 외부 검증으로 해석하지 않습니다.",
        "- 반복적으로 같은 개발·고정 평가에 접근하면 간접 과적합이 쌓입니다. 실패 후보를 포함한 모든 실행을 보관하고, 독립된 신규 피험자·세션을 수집해야 합니다.",
        "- 창은 같은 기록·사람 안에서 상관되어 있습니다. 창 수를 독립 표본 수로 해석하지 않으며 불확실성은 피험자 단위로 다룹니다.",
        "- 정적 CSV 자세 분류만 확인합니다. 실제 센서 취득, 의자·착석 위치 변화, 재착석, 빈 의자, 자세 전환, 패키지 앱·물리 하드웨어는 이 결과로 검증되지 않습니다.",
        ("- 공통 평가 창은 과거 예약 기록 제외 정책을 유지합니다. 복원 정책을 명시한 기준·후보는 학습 피험자의 일반 적격 정자세 기록을 추가하며, 개인 등록·교정 기록과 평가 피험자의 추가 기록은 사용하지 않습니다."
         if p.get("supplemental_training") else "- 이번 동결 코호트는 등록용 기록을 전체 비교군에서 제외한 역사적 정책을 유지합니다. 현재 앱의 데이터 적격성 정책과 다를 수 있으므로 새 코호트와 점수를 섞지 않습니다."),
        "- `decision_softmax`의 값은 정규화된 결정점수이며 신뢰도로 보정된 확률이 아닙니다. 확률 평균 결합의 적합성은 별도 평가가 필요합니다.", "",
        "- 시계열 후보와 통계 기준은 동일하게 동결한 보간 입력을 사용합니다. 현재 전체 기록 보간은 창 경계 뒤의 원시 표본을 참조할 수 있어 엄격한 실시간 인과성을 보장하지 않습니다. 이 반복은 해당 동결 프로토콜의 후향적 비교이며 인과적 창 생성 변경은 별도 실험이 필요합니다.", "",
        ("다음 반복은 사용자 채택한 이번 구성에서 시작하며 이전 기준 모델을 별도 보존합니다. " if adopted else
         "다음 반복은 검증된 개선이 있을 때 그 구성에서 시작하고, 없으면 이번 기준을 유지합니다. ") +
        "실패 구성을 단순히 재탐색하지 않고 아래의 독립된 소규모 가설을 사전 등록한 뒤 병렬 비교합니다.", "",
        *next_experiments(candidates),
        "4. 잠긴 신규 사용자·새 세션의 측정 계획을 준비합니다. 현재 자료에서의 소폭 상승이 반복 실패하면 새 모집단·센서 배치 증거를 우선합니다.", "",
        "## 완전한 파이프라인과 재현 절차", "",
        "통계 모델은 596차원 입력, 시계열 모델은 같은 평가 행의 36×16 센서 배열을 받습니다. 전처리·특징 선택·모델 파라미터·가중치·후처리는 직렬화된 추정기에 포함됩니다. "
        "공통 창 생성과 특징 이름 구현도 source_snapshot에 고정되어 있습니다. base_method 표시는 출처이며 실제 하이퍼파라미터는 model_params가 결정합니다.", ""]
    for row in candidates:
        lines += [f"### `{row['id']}` 구성", ""] + code(row["pipeline"])
    lines += ["### 동결 소스에서 재실행", "",
        "완료된 실행 폴더는 덮어쓰지 않습니다. 아래 명령은 동결 소스와 입력을 읽고 모든 개발 후보를 임시 폴더에서 다시 학습합니다. "
        "원 실행이 고정 평가까지 진행된 경우에만 이미 봉인된 최종 후보의 고정 평가를 재현합니다. "
        "이 재실행은 새로운 후보 선택이나 독립 검증이 아닙니다.", ""]
    snapshot = out / "source_snapshot"
    rerun = f"""cd {shlex.quote(str(snapshot))}
POSTURE_RESEARCH_ROOT={shlex.quote(str(ROOT))} {shlex.quote(str(ROOT / '.venv/bin/python'))} - <<'PY'
import json
import tempfile
from pathlib import Path
from experiments.iterative_improvement.run import verify, load_data, indices, fit_one

out = Path({str(out)!r})
protocol = verify(out)
data = load_data(out)
selection = json.loads((out / 'selection_lock.json').read_text())
evaluation = json.loads((out / 'evaluation.json').read_text())
destination = Path(tempfile.mkdtemp(prefix='posture-checkpoint-reproduce-'))
recipes = {{'baseline': protocol['baseline'], **{{row['id']: row['pipeline'] for row in selection['candidates']}}}}
for name, pipeline in recipes.items():
    for part in protocol['partitions']:
        train, valid = indices(data, part)
        result = fit_one(out, data, pipeline, train, valid,
            destination / 'development' / name / part['id'], protocol['seeds'][0])
        print(name, part['id'], result['metrics']['macro_f1'])
if evaluation.get('holdout_evaluated'):
    selected = selection['selected']
    train, valid = indices(data, protocol['partitions'][0], final=True)
    for seed in protocol['seeds']:
        for name in ('baseline', selected['id']):
            result = fit_one(out, data, recipes[name], train, valid,
                destination / 'final' / str(seed) / name, seed)
            print(name, seed, result['metrics']['macro_f1'])
print(destination)
PY"""
    lines += code(rerun, "sh")
    lines += ["재현에는 다음 실행환경의 동일 패키지 버전과 Python ABI가 필요합니다. 모델 직렬화 파일은 신뢰할 수 있는 자체 생성물만 로드합니다.", "",
        "## 실행 이력·출처·무결성", "",
        f"소스 리비전: `{p['source_revision']}`. 추적 중인 수정과 신규 파일 목록까지 기록했으며 커밋만으로 소스를 재현하지 않습니다. "
        "실제 실행은 해시로 고정한 source_snapshot을 기준으로 합니다.", "",
        "### 실행 명령", ""] + code(p["command"])
    lines += ["### 실행 당시 작업 트리 상태", ""] + code(p.get("dirty_state") or "(clean)", "text")
    lines += table(["입력·봉인 파일", "SHA-256"], [
        ("protocol.json", protocol_hash), ("selection_lock.json", selection_hash),
        ("dataset.json", p["dataset_sha256"]), ("data.npz", p["cache_sha256"]),
        ("evaluation.json", digest(out / "evaluation.json")),
        *([("reproducibility.json", digest(out / "reproducibility.json"))] if reproduction else [])])
    lines += ["개별 원본 입력 경로·해시·크기·mtime은 데이터 명세와 보호 스냅샷에 있습니다. "
        "보호 스냅샷 자체의 존재가 실행 후 불변성을 입증하지는 않습니다. 원본 보호 검증은 별도 verify 명령으로 확인합니다.", ""]
    lines += code(f"{shlex.quote(str(ROOT / '.venv/bin/python'))} -m experiments.iterative_improvement.run --output {shlex.quote(str(out))} --phase verify", "sh")
    lines += ["### 동결 소스 해시", ""]
    lines += table(["소스", "SHA-256"], [(name, value) for name, value in sorted(p["source_hashes"].items())])
    lines += ["### 패키지 환경", ""] + code("\n".join(p["environment"]), "text")
    lines += ["### 근거 파일", ""]
    for filename, label in (("protocol.json", "실행 프로토콜"), ("protocol_lock.json", "프로토콜 봉인"),
            ("dataset.json", "입력·분할·개별 원본 해시"), ("protected_before.json", "원본·기존 모델 보호 스냅샷"),
            ("selection_lock.json", "후보 선택과 전체 개발 요약"), ("selection_seal.json", "선택 봉인"),
            ("evaluation.json", "최종 채택 결정"), ("reproducibility.json", "재현성 근거"),
            ("combination_lock.json", "결합 구성"),
            ("audits/structured_fitted_identity.json", "실제 TRAIN·저장 학습 상태 독립 감사"),
            ("audits/source_validation.json", "소스 검증 결과와 검증 범위"),
            ("audits/runtime.json", "소스 런타임 CSV 검사")):
        path = out / filename
        if path.is_file():
            lines.append("- " + link(report, path, label))
    for directory in ("development", "robustness", "final", "reproduction", "logs", "source_snapshot", "audits"):
        if (out / directory).is_dir():
            lines.append("- " + link(report, out / directory, directory))
    lines += ["", ("이번 체크포인트는 사용자 채택 상태와 요청 근거를 명시하여 등록합니다. " if adopted else
        "탈락 후보와 예측·설정·모델은 감사 가능한 실험 이력으로 보관하지만 검증 체크포인트 등록부에는 추가하지 않습니다. ") +
        "검증에 성공한 체크포인트는 위 보고서와 모델을 포함한 완전한 근거 파일의 해시를 등록해 모델 추가 모드에서 재사용합니다.", ""]
    findings = out / "audits/report_findings.md"
    if findings.is_file():
        lines += ["## 평가 후 독립 감사와 해석", "", findings.read_text(encoding="utf-8"), "",
            link(report, findings, "감사·해석 원문") + f" · SHA-256 `{digest(findings)}`", ""]
    return "\n".join(lines)


def write_report(out: Path, report: Path, *, checkpoint_name=None, checkpoint_version="1.0.0"):
    out, report = Path(out).resolve(), Path(report).absolute()
    content = render(out, report, checkpoint_name, checkpoint_version)
    if report.exists():
        if report.read_text(encoding="utf-8") == content:
            return report
        raise ValueError("기존 보고서는 불변입니다. 다른 파일명으로 새 보고서를 작성하세요.")
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("x", encoding="utf-8") as stream:
        stream.write(content)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint-name")
    parser.add_argument("--checkpoint-version", default="1.0.0")
    args = parser.parse_args()
    path = write_report(args.output, args.report, checkpoint_name=args.checkpoint_name,
        checkpoint_version=args.checkpoint_version)
    print(json.dumps({"report": str(path), "sha256": digest(path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
