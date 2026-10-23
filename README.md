# seongju_melon_detector

## 실시간 카메라 프레임 전달

`camera_shm`은 최신 완료 프레임 하나만 유지합니다. 생산자의 쓰기와 소비자의
복사는 프로세스 간 잠금으로 보호하며, `timestamp`, RGB, Depth, `sequence_id`를
같은 잠금 안에서 갱신하고 읽습니다. 소비자는 복사 이후 잠금 밖에서 인식합니다.

```python
last_sequence_id = 0
# 각 인식 반복에서 호출합니다. 세마포어를 사용하는 경우 먼저 알림을 기다립니다.
frame = camera_shm.read(last_sequence_id=last_sequence_id)
if frame is not None:
    last_sequence_id = frame.sequence_id
    # frame.color / frame.depth는 독립된 복사본입니다. 여기서 인식합니다.
```

소비자가 느리면 중간 프레임은 건너뜁니다. 프레임 큐는 없으며, 카메라 러너는
세마포어 알림도 최대 하나만 대기시킵니다. 알림은 새 데이터 확인을 위한 힌트이고,
실제 중복 여부는 복사본의 `sequence_id`로 확인합니다. 공유 세마포어는 여러
소비자 모두에게 전달되는 방송 알림이 아닙니다.

30 FPS의 프레임 간격은 약 33.3ms입니다. 소비자의 복사 중에는 생산자가 잠깐
기다릴 수 있습니다. 실제 장비에서 잠금 대기·복사 시간과 촬영부터 인식 결과까지의
지연을 확인해야 하며, SDK 내부 버퍼나 인식 작업 큐의 지연까지 제거하는 구조는 아닙니다.

공유 메모리 레이아웃이 단일 버퍼로 변경되었으므로 기존 프로세스를 모두 종료하고,
`main.py`, 카메라 러너, 소비자 순서로 다시 실행해야 합니다.

## RGB / Depth / flange 데이터 수집

각 명령을 별도 터미널에서 저장소 루트 기준으로 실행합니다.

```bash
.venv/bin/python main.py
.venv/bin/python -m camera.camera_runner
.venv/bin/python training/data_collection.py
```

로봇 flange 송신 프로세스도 기존 방식으로 실행해야 합니다. 수집기는 카메라를
직접 열지 않습니다. 미리보기는 `camera_shm`의 최신 영상을 사용합니다.

- `SPACE`: 수집 시작 / 일시정지. 일시정지 전에 Queue에 들어온 데이터는 계속 저장합니다.
- `q`: 생산을 중지하고 Queue에 남은 데이터를 저장한 뒤 종료합니다.
  수집기를 먼저 종료한 후 camera runner와 main을 종료하세요.
- 저장 위치는 `training/data_collection.py`의 `SAVE_PATH`입니다.
  현재 `/media/uon/jsh/melon_data`이며, USB 마운트 경로가 미리 존재해야 합니다.
- `rgb/`, `depth/`, `flange/`에 동일한 파일명으로 RGB PNG, uint16 Depth PNG,
  flange JSON을 저장합니다. JSON에는 이미지 timestamp, 가장 가까운 flange의
  timestamp, 시간 차이(µs), 원본 4×4 TF가 들어갑니다. flange가 비활성이거나
  데이터가 없으면 `null`로 기록하며 이미지는 저장합니다.
- 화면 변화/선명도로 저장을 제한하지 않습니다. 수집 실행 내 같은 이미지 timestamp는
  한 번만 저장하고, 선명도는 파일명의 `sharp` / `blur` 태그로만 표시합니다.
- camera runner는 복사본을 최대 8개 보관합니다(1280×720 RGB+Depth 약 35 MiB).
  수집기는 최대 2개의 프레임 저장을 병렬 처리하며 전송과 저장을 겹칩니다.
  저장 작업의 대기 수도 제한하므로 메모리 사용량이 무한히 늘지 않습니다.
  Queue가 가득 차면 새 수집 프레임을 버리고 제어용 shared memory 갱신은 계속합니다.
  화면의 `Dropped`는 runner 실행 이후 누적 값이고, 종료 시
  `collection_*.json`에 수집 세션의 Queue 누락 수와 저장 수를 기록합니다.
  이 수치는 카메라 SDK 내부의 누락을 포함하지 않습니다.
  `Write`는 프레임 쌍당 평균 저장 시간이며, 세션 JSON에도 평균/최대 저장 시간(ms)을
  기록합니다. 병렬화 후에도 저장 장치의 지속 처리량이 부족하면 누락될 수 있습니다.
- flange와 카메라 timestamp가 동일한 시간 기준이어야 매칭 시간 차이가 의미 있습니다.
  TF 단위와 좌표계는 flange shared memory의 원본을 그대로 유지합니다.

실제 30 FPS 지속 저장과 시간 동기화는 카메라·로봇·저장 장치에서 확인해야 합니다.
