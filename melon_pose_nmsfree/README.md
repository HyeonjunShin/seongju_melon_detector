# TensorRT serving

현재 카메라용 추론 파일과 엔진 재빌드용 ONNX를 포함한다. 원본은
`/media/uon/Samsung USB/melon_pose_nmsfree`에 그대로 보존되어 있다.

```text
melon_pose_nmsfree/
├── __init__.py
├── detector.py          # Detector API 및 이미지 추론 CLI
├── trt_runtime.py       # TensorRT 로딩·GPU 버퍼·실행
├── tensorrt_deploy.py   # 학습 모듈 없이 ONNX → 엔진 빌드
├── models/
│   ├── rtx4090_fp16.engine
│   ├── rtx4090_fp16.json
│   ├── rtx5060_laptop_fp16.engine  # 현재 PC에서 빌드·추론 확인
│   ├── rtx5060_laptop_fp16.json
│   ├── model.onnx       # 848×530 통합 모델, 새 GPU용 엔진 빌드 입력
│   └── model.json       # ONNX 입력·출력 규격 및 체크섬
├── requirements.txt
├── THIRD_PARTY.md
└── README.md
```

엔진 메타데이터의 환경은 RTX 4090, TensorRT `10.14.1.48.post1`,
PyTorch `2.12.0+cu130`이다. CUDA에 맞는 PyTorch 설치가 필요하다.
엔진은 GPU와 TensorRT 버전을 검사한다. 다른 환경에서 재빌드할 때는
이 폴더의 `models/model.onnx`와 `tensorrt_deploy.py`를 사용한다.

프로젝트 루트에서 현재 PC용 엔진 빌드:

```bash
.venv/bin/python -m melon_pose_nmsfree.tensorrt_deploy build \
  --onnx melon_pose_nmsfree/models/model.onnx \
  --engine melon_pose_nmsfree/models/rtx5060_laptop_fp16.engine
```

실행할 NVIDIA GPU가 보이는 환경에서 실행한다. 기존 엔진이나 메타데이터는
덮어쓰지 않으므로 다시 빌드할 때는 새 출력 이름을 지정한다.

프로젝트 루트에서:

```python
import cv2
from melon_pose_nmsfree import Detector

detector = Detector("melon_pose_nmsfree/models/rtx5060_laptop_fp16.engine")

# Detector는 루프 밖에서 한 번 생성. frame은 CameraShm.read() 반환값.
objects = detector(cv2.cvtColor(frame.color, cv2.COLOR_RGB2BGR))
```

입력은 `(530, 848, 3)`의 `uint8` BGR이다. 전처리·후처리는 엔진에 포함된다.
출력 좌표는 원본 이미지 픽셀, 방향각은 도 단위다. 깊이 입력은 사용하지 않는다.
동일 Detector에 대한 호출은 순차적으로 수행한다.

RTX 5060 Laptop GPU에서 새 엔진의 로딩과 빈 프레임·USB 예시 이미지 추론을
확인했다. 예시 이미지에서는 3개 객체가 반환되었다. 정확도 평가나 실시간
카메라 처리 속도 검증은 수행하지 않았다.

카메라 SHM을 소비하는 검출 미리보기는 프로젝트 루트에서 실행한다.
`main.py`와 `camera_runner.py`가 먼저 실행되어 있어야 한다.

```bash
.venv/bin/python -m detector.xpu_detection
```

처리속도 원인을 확인할 때는 기존 검출 프로그램을 종료한 뒤 화면 없이 비교한다.

```bash
.venv/bin/python -m detector.xpu_detection --no-display
```

두 모드 모두 1초마다 FPS와 `wait`(프레임 대기), `shm`(복사), `infer`(전송 포함 추론),
`draw`(그리기), `gui`(창 표시·이벤트 처리)의 평균 ms를 출력한다.
`skipped`는 첫 프레임 이후 sequence가 건너뛴 수이며 처리 누적 큐의 길이가 아니다.
화면 없는 모드는 Ctrl+C로 종료한다.

현재 PC의 엔진이 기본값이다. `--engine`으로 다른 엔진을 지정할 수 있고,
`--confidence`는 엔진 내부 임계값인 0.3 이상으로 지정한다. 기존과 같이
매 프레임 최고 신뢰도 객체 하나를 선택하고 추론 결과의 박스·Body·Knot를 그대로 표시한다.
키포인트는 Body, Knot 순서이고 가시성은 확률이다. 종료는 Q 또는 ESC이다.
