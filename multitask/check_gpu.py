import torch

print("="*50)
print("GPU 사용 가능 여부 체크")
print("="*50)
print(f"PyTorch 버전: {torch.__version__}")
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA 버전: {torch.version.cuda}")
    print(f"GPU 개수: {torch.cuda.device_count()}")
    print(f"GPU 이름: {torch.cuda.get_device_name(0)}")
    print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print("\n✅ GPU를 사용할 수 있습니다!")
else:
    print("\n❌ GPU를 사용할 수 없습니다. CPU로 학습됩니다.")
    print("CUDA가 설치되지 않았거나 PyTorch가 CPU 버전으로 설치되었습니다.")
print("="*50)
