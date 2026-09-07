import matplotlib.pyplot as plt
import librosa.display

from preprocessing.audio import load_audio, preprocess_audio
from preprocessing.features import extract_mel_spectrogram


audio_path = "deepfake_detector/data/test_audio.wav"

audio, sample_rate = load_audio(audio_path)

audio = preprocess_audio(audio)

mel = extract_mel_spectrogram(
    audio,
    sample_rate
)

print("================================")
print("VoxShield Mel Spectrogram Test")
print("================================")

print("Audio samples:", len(audio))
print("Sample rate:", sample_rate)
print("Mel spectrogram shape:", mel.shape)

plt.figure(figsize=(10, 4))

librosa.display.specshow(
    mel,
    sr=sample_rate,
    hop_length=256,
    x_axis="time",
    y_axis="mel"
)

plt.colorbar(format="%+2.0f dB")
plt.title("Mel Spectrogram")
plt.tight_layout()
plt.show()