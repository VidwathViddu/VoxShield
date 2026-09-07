import librosa
import numpy as np


def extract_mel_spectrogram(
    audio,
    sample_rate=16000,
    n_mels=128,
    n_fft=1024,
    hop_length=256
):
    """
    Convert an audio waveform into a log-Mel spectrogram.
    """

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    return mel_db