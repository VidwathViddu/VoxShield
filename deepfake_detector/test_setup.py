import torch
import torchaudio
import librosa

print("==============================")
print("VoxShield - Member 1")
print("==============================")

print("PyTorch:", torch.__version__)
print("Torchaudio:", torchaudio.__version__)
print("Librosa:", librosa.__version__)

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("Running on CPU")