"""한국어 안내와 영문 전문용어 표시 문구. 저장된 식별자와 원본 진단 기록은 변경하지 않는다."""
import ast
import json
import re

WORDS = {
    "rbf_svm": "RBF SVM", "RBF-SVM": "RBF SVM",
    "RBF SVM": "RBF SVM", "logistic_regression": "Logistic Regression",
    "Logistic Regression": "Logistic Regression", "random_forest": "Random Forest",
    "Random Forest": "Random Forest", "extra_trees": "Extra Trees", "Extra Trees": "Extra Trees",
    "catboost": "CatBoost", "xgboost": "XGBoost", "minirocket": "MiniRocket + Logistic Regression",
    "cnn1d": "Small 1D-CNN", "mlp": "Small MLP",
    "Soft Voting": "Soft Voting", "baseline_required": "빈 좌석 확인 필요",
    "baseline_collecting": "빈 좌석 기준 측정 중", "profile_required": "정자세 등록 필요",
    "registration_complete": "정자세 등록 완료", "profile_loaded": "정자세 기준 불러옴", "registering": "정자세 등록 중",
    "correct": "정자세", "incorrect": "잘못된 자세", "classifying": "판단 중",
    "uncertain": "판단 유보", "movement": "움직임 감지", "insufficient": "입력 부족",
    "empty": "빈 좌석", "empty_seat": "빈 좌석", "input_error": "입력 오류",
    "invalid_input": "입력 오류", "disconnected": "센서 연결 끊김", "session_stopped": "측정 종료",
    "deployment": "배포 모드", "development": "개발 모드", "completed": "완료",
    "running": "진행 중", "cancelled": "취소됨", "failed": "실패", "pending": "대기",
    "passed": "통과", "pass": "통과", "fail": "실패", "warning": "주의",
    "validation": "검증", "final_evaluation": "최종 평가", "train": "학습", "loso": "LOSO",
    "none": "없음", "None": "없음", "True": "예", "False": "아니요",
    "C": "C", "gamma": "gamma", "scale": "scale",
    "n_estimators": "n_estimators", "min_samples_leaf": "min_samples_leaf",
    "max_features": "max_features", "sqrt": "sqrt", "balanced": "balanced",
    "completed_with_failures": "완료 (일부 매개변수 실패)", "not_checked": "미검사",
    "usable": "사용 가능", "environment_unavailable": "실행 환경 사용 불가",
    "rejected": "승인 거절", "revoked": "폐기됨",
}

PHRASES = {
    "Posture is correct.": "정자세입니다.",
    "Deviated from reference posture · Verifying posture type.": "기준 정자세에서 벗어났습니다 · 자세 유형 확인 중",
    "Classifying · Movement detected.": "판단 중 · 움직임이 감지되었습니다.",
    "Classifying · Insufficient input.": "판단 중 · 입력 데이터가 부족합니다.",
    "Classifying": "판단 중",
    "Empty seat · Classification and alerts stopped.": "빈 좌석 · 자세 판단과 알림을 중지했습니다.",
    "Input error · Classification and alerts stopped.": "입력 오류 · 자세 판단과 알림을 중지했습니다.",
    "Register or load a correct-posture reference for this user.": "사용자의 정자세 기준을 등록하거나 불러오세요.",
    "Return to your registered upright posture, uncross your legs, and place both feet flat on the floor.": "등록한 바른 자세로 돌아오세요. 꼬인 다리를 풀고 양발을 바닥에 편안하게 놓으세요.",
    "Gently bring your torso upright and support your back.": "상체를 천천히 세우고 등을 편안하게 받치세요.",
    "Bring your torso back to a comfortable upright position.": "상체를 편안한 바른 자세로 되돌리세요.",
    "Bring your weight back toward the center of the seat.": "몸의 무게를 좌석 중앙으로 고르게 옮기세요.",
    "Uncross your left leg and place both feet flat on the floor.": "꼬인 왼쪽 다리를 풀고 양발을 바닥에 놓으세요.",
    "Uncross your right leg and place both feet flat on the floor.": "꼬인 오른쪽 다리를 풀고 양발을 바닥에 놓으세요.",
    "Leave the seat empty, then select Check empty seat.": "좌석을 비운 뒤 ‘빈 좌석 확인’을 누르세요.",
    "Keep the seat empty and still for 3 seconds.": "좌석을 비운 상태로 3초 동안 움직이지 마세요.",
    "Check the empty seat first, then sit down before registration.": "빈 좌석을 먼저 확인한 뒤 앉아서 정자세를 등록하세요.",
    "Maintain the instructed correct posture for 10 seconds.": "안내에 따라 정자세를 10초 동안 유지하세요.",
    "Sensor input unavailable. Classification and alerts are paused.": "센서 입력을 받을 수 없습니다. 자세 판단과 알림을 중지했습니다.",
    "The empty-seat reading was unstable. Repeat the check.": "빈 좌석 측정값이 불안정합니다. 다시 확인하세요.",
    "No saved reference for this user.": "이 사용자의 저장된 정자세 기준이 없습니다.",
    "The baseline changed. Re-register the correct posture.": "빈 좌석 기준이 바뀌었습니다. 정자세를 다시 등록하세요.",
    "Personal reference loaded. Sit down to begin.": "사용자의 정자세 기준을 불러왔습니다. 좌석에 앉으면 시작합니다.",
    "Register the correct posture:": "정자세를 등록하세요:",
    "Registering correct posture:": "정자세 등록 중:",
    "Reference saved. Personal posture guidance is active.": "정자세 기준을 저장했습니다. 개인별 자세 안내를 시작합니다.",
    "Registration failed. Repeat the measurement:": "등록하지 못했습니다. 다시 측정하세요:",
    "The seat is empty.": "좌석이 비어 있습니다.",
    "Sensor values must be finite numeric values.": "센서값은 유한한 숫자여야 합니다.",
    "Raw pressure sensor values must be nonnegative.": "압력 센서값은 음수일 수 없습니다.",
    "Personal calibration requires the ordered 4x4 pressure channels.": "개인 보정에는 순서가 일치하는 4행 4열 압력 센서가 필요합니다.",
    "Sensor configuration, chair, and placement identifiers are required.": "센서 구성, 의자, 설치 위치 식별자를 입력하세요.",
    "Unknown pressure direction.": "압력 변화 방향 설정을 확인하세요.",
    "Calibration timing and stability settings must be positive and finite.": "등록 시간과 안정성 설정은 유한한 양수여야 합니다.",
    "Unsupported calibration version; register the reference again.": "지원하지 않는 보정 버전입니다. 정자세를 다시 등록하세요.",
    "A user identifier is required.": "사용자 식별자를 입력하세요.",
    "Invalid calibration profile version; register again.": "프로필 버전이 올바르지 않습니다. 다시 등록하세요.",
    "Registration time must include a timezone.": "등록 시각에는 시간대 정보가 필요합니다.",
    "Invalid calibration registration time.": "보정 등록 시각이 올바르지 않습니다.",
    "Correct-posture confirmation and a separate empty-seat check are required.": "정자세 안내 확인과 별도의 빈 좌석 확인이 필요합니다.",
    "Invalid pressure variation range.": "압력 변동 범위가 올바르지 않습니다.",
    "Invalid personal pressure ratios or allowable deviations.": "개인 압력 비율 또는 허용 편차가 올바르지 않습니다.",
    "Missing calibration stability metrics.": "보정 안정성 지표가 누락되었습니다.",
    "Invalid calibration stability metrics.": "보정 안정성 지표가 올바르지 않습니다.",
    "Personal ratio tolerances do not match the recorded registration variation.": "압력 비율 허용 범위가 등록 기록의 변동과 일치하지 않습니다.",
    "The calibration profile does not meet its registration stability requirements.": "프로필이 정자세 등록 안정성 기준을 충족하지 않습니다.",
    "Sensor configuration, chair, or placement changed; register the reference again.": "센서 구성, 의자 또는 설치 위치가 바뀌었습니다. 정자세를 다시 등록하세요.",
    "Invalid calibration profile; register again:": "프로필이 올바르지 않습니다. 다시 등록하세요:",
    "Select a user before registering a correct posture.": "정자세를 등록하기 전에 사용자를 선택하세요.",
    "Confirm the posture guidance and perform the separate empty-seat check first.": "자세 안내를 확인하고 빈 좌석 확인을 먼저 진행하세요.",
    "Repeat the correct-posture registration.": "정자세 등록을 다시 진행하세요.",
    "This registration is complete; start a new session to re-register.": "등록이 완료되었습니다. 재등록하려면 새 등록을 시작하세요.",
    "Movement, an empty seat, or invalid sensor input was detected.": "움직임, 빈 좌석 또는 잘못된 센서 입력이 감지되었습니다.",
    "Invalid sensor timestamp.": "센서 측정 시각이 올바르지 않습니다.",
    "Sensor timestamps are out of order or contain a gap.": "센서 측정 시각의 순서가 잘못되었거나 입력이 끊겼습니다.",
    "Insufficient pressure above the empty-seat baseline.": "빈 좌석 기준보다 높은 압력이 충분히 감지되지 않았습니다.",
    "Pressure distribution moved during registration.": "등록 중 압력 분포가 변했습니다.",
    "Total pressure moved during registration.": "등록 중 전체 압력이 변했습니다.",
    "The recorded posture was not stable enough.": "측정한 자세가 충분히 안정적이지 않습니다.",
    "A different profile already uses this version identifier.": "다른 프로필에서 이미 사용 중인 버전 식별자입니다.",
    "The stored reference belongs to a different user.": "저장된 정자세 기준의 사용자가 다릅니다.",
    "Cannot use the saved reference; register again:": "저장된 기준을 사용할 수 없습니다. 다시 등록하세요:",
    "Alert settings require a known mode and positive finite timings.": "알림 모드와 시간을 확인하세요. 시간은 유한한 양수여야 합니다.",
    "Alert mode must be development or deployment.": "개발 모드 또는 배포 모드를 선택하세요.",
    "Select at least one supported model:": "학습할 모델을 하나 이상 선택하세요:",
    "At least four subjects are required for train/validation/final evaluation.": "학습·검증·최종 평가를 분리하려면 최소 4명의 데이터가 필요합니다.",
    "Training output already exists. Use a new run directory.": "학습 결과가 이미 있습니다. 새 실행 폴더를 사용하세요.",
    "Model comparison cancelled": "모델 비교를 취소했습니다",
    "Every model failed; inspect the comparison failure reasons.": "모든 모델 학습이 실패했습니다. 비교 결과에서 실패 원인을 확인하세요.",
    "Comparison incomplete; no valid candidate for ": "모델 비교 미완료 · 유효한 후보가 없는 모델: ",
    "Loading and auditing shared dataset": "공통 데이터 불러오기 및 품질 확인",
    "Selection LOCKED before final evaluation:": "최종 평가 전 모델 선택 확정:",
    "validation Macro F1": "검증 Macro F1", "final Macro F1": "최종 Macro F1",
    "FAILED:": "실패:", "parameters=": "매개변수=", "subjects": "명", "records": "기록", "windows": "분석 창",
    "No saved model activation history is available.": "저장된 모델 적용 이력이 없습니다.",
    "Invalid model activation history.": "모델 적용 이력이 올바르지 않습니다.",
    "No previously active model still passes approval and validation.": "승인과 검증을 통과한 이전 모델이 없습니다.",
    "Return to your registered upright posture.": "등록한 바른 자세로 돌아오세요.",
    "Duplicate sensor recording:": "중복 센서 기록:",
    "Unsupported model family:": "지원하지 않는 모델 종류:",
    "lacks classes:": "분할에 누락된 자세:",
    "ms/window": "ms/분석 창", "bytes": "바이트",
}


def ko(value):
    """Known application text only; paths/user identifiers remain literal."""
    if isinstance(value, json.JSONDecodeError):
        return f"저장 파일 형식이 올바르지 않습니다. {value.lineno}행 {value.colno}열을 확인하세요."
    if isinstance(value, FileNotFoundError):
        return f"파일을 찾을 수 없습니다: {value.filename or '경로를 확인하세요.'}"
    if isinstance(value, PermissionError):
        return f"파일 접근 권한이 없습니다: {value.filename or '접근 권한을 확인하세요.'}"
    text = str(value)
    if text in WORDS:
        return WORDS[text]
    for source, translated in sorted(PHRASES.items(), key=lambda item: -len(item[0])):
        text = text.replace(source, translated)
    for source in ("Logistic Regression", "Random Forest", "Extra Trees", "RBF-SVM", "RBF SVM", "Soft Voting"):
        text = text.replace(source, WORDS[source])
    text = re.sub(r"Expected (\d+) finite sensor values\.", r"유한한 센서값 \1개가 필요합니다.", text)
    text = re.sub(r"Registration needs at least ([\d.]+) seconds and (\d+) valid samples\.", r"등록에는 최소 \1초와 유효한 표본 \2개가 필요합니다.", text)
    text = re.sub(r"\b(train|validation|final_evaluation|loso|rbf_svm|logistic_regression|random_forest|extra_trees|catboost|xgboost|minirocket|cnn1d|mlp)\b", lambda m: WORDS[m[0]], text)
    text = re.sub(r"(\d+(?:\.\d+)?)s\b", r"\1초", text)
    text = re.sub(r"could not convert string to float:.*|invalid literal for int\(\).*", "숫자 입력란에 올바른 숫자를 입력하세요.", text)
    match = re.search(r"매개변수=(\{.*\})", text)
    if match:
        try:
            text = text[:match.start(1)] + parameters_ko(ast.literal_eval(match[1])) + text[match.end(1):]
        except (ValueError, SyntaxError):
            pass
    return text


def parameters_ko(parameters):
    return ", ".join(f"{ko(key)}: {ko(value)}" for key, value in (parameters or {}).items()) or "없음"
