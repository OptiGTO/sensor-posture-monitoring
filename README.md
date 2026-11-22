# 자세 측정 및 관리

좌판의 4×4 압력 센서에서 들어오는 16채널 시계열로 착석 자세 7종을 분류하는 Python/Tk 프로젝트입니다. 정자세, 앞으로 숙이기, 뒤로 기대기, 왼쪽·오른쪽 기울임, 왼쪽·오른쪽 다리 꼬기를 다룹니다. 통합 앱, CSV 재생·추론, 모델 학습·평가 코드를 제공합니다.

## 실행

Python과 Tkinter가 설치된 환경에서 저장소 루트로 이동해 실행합니다.

~~~sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements_core.txt
.venv/bin/python software/posture_app.py
~~~

XGBoost, LightGBM, CatBoost, PyTorch 등 확장 모델을 사용하려면 **requirements.txt**의 의존성을 설치합니다.

## 데이터로 학습하고 재생하기

별도로 확보한 기록을 피험자 디렉터리별로 정리합니다. 기록의 CSV 열은 **elapsed_s,label,R11,...,R44** 순서입니다. **label**에는 사람이 지정한 실제 자세를 기록합니다. 시간과 채널의 처리 방식은 [CSV 로더](src/posture_core.py)에서 확인할 수 있습니다.

~~~sh
.venv/bin/python train_window_models.py --data /path/to/labeled-data --output-dir ./results/my-run --windows 3 5
.venv/bin/python inference_window.py --input /path/to/recording.csv --model-dir ./results/my-run/artifacts
~~~

학습 결과의 모델 파일을 추론 명령의 **--model-dir**에 지정합니다. 앱에서 자세를 예측하려면 검증·승인을 거쳐 활성화한 모델이 필요합니다. 앱의 모델 검증에는 별도로 확보한 기준 기록을 **example/sample_input.csv**에 지정합니다. 피험자 측정 원본, 학습된 모델, 모델 검증용 기준 CSV와 예측값, 설치용 번들 및 내부 연구 보고서는 비공개로 관리합니다. 기존 연구 재현 검사와 macOS 번들 빌드에는 별도의 입력 자료가 필요합니다.

## 테스트

다음 테스트는 자체 설정과 합성 입력으로 실행할 수 있습니다.

~~~sh
export POSTURE_APP_STATE_ROOT="$(mktemp -d)"
.venv/bin/python -m unittest discover -s tests -p 'test_training_deduplication.py'
.venv/bin/python -m unittest discover -s tests -p 'test_app_paths.py'
~~~

## 라이선스

프로젝트 소스는 [MIT 라이선스](LICENSE)로 공개합니다. 포함된 [sktime MiniRocket 코드](src/_vendor/README.md)에는 [BSD 3-Clause 라이선스](src/_vendor/LICENSE.sktime)가 적용됩니다.
