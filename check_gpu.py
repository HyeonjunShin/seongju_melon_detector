import torch

# 1. GPU 사용 가능 여부 확인 (True가 나와야 GPU 사용 가능)
print("GPU 사용 가능 여부:", torch.cuda.is_available())

# 2. 연결된 GPU 개수 확인
print("연결된 GPU 개수:", torch.cuda.device_count())

# 3. 현재 사용 중인 GPU 장치 인덱스 확인
print("현재 GPU 인덱스:", torch.cuda.current_device())

# 4. GPU 모델명 확인 (0번 장치 기준)
if torch.cuda.is_available():
    print("GPU 모델명:", torch.cuda.get_device_name(0))
