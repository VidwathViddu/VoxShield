import librosa
import numpy as np

TARGET_SAMPLE_RATE = 16000

def load_audio(audio_path):
    audio, sample_rate = librosa.load(
        audio_path,
        sr=TARGET_SAMPLE_RATE,
        mono=True
    )
    return audio, sample_rate

def normalize_audio(audio):
    max_amplitude = np.max(np.abs(audio))
    if max_amplitude > 0:
        audio = audio / max_amplitude
    return audio

def preprocess_audio(audio):
    audio = normalize_audio(audio)
    return audio