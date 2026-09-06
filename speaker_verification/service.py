"""Extract ECAPA-TDNN speaker embeddings and compare them with cosine similarity.

This module verifies *speaker identity*.  It does not determine whether an
audio recording is human or synthetically generated.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torchaudio
import soundfile as sf
from speechbrain.inference.speaker import SpeakerRecognition

MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"
SAMPLE_RATE = 16_000
DEFAULT_THRESHOLD = 0.65


class SpeakerVerifier:
    """Pretrained ECAPA-TDNN embedding extractor and verifier.

    ``threshold`` is deliberately configurable: it must be calibrated from
    validation recordings collected for the actual deployment environment.
    """

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        model_dir: str | Path = "pretrained_models/spkrec-ecapa-voxceleb",
        device: str | None = None,
    ) -> None:
        if not -1.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between -1.0 and 1.0")
        self.threshold = threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SpeakerRecognition.from_hparams(
            source=MODEL_ID,
            savedir=str(model_dir),
            run_opts={"device": self.device},
        )

    @staticmethod
    def _load_audio(audio_path: str | Path) -> torch.Tensor:
        """Load an audio file as a single-channel 16 kHz waveform."""
        data, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
        waveform = torch.from_numpy(data.T)
        if waveform.numel() == 0:
            raise ValueError(f"Audio file is empty: {audio_path}")
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sample_rate, SAMPLE_RATE)
        return waveform

    @torch.inference_mode()
    def embedding(self, audio_path: str | Path) -> torch.Tensor:
        """Return a unit-normalized 192-dimensional speaker embedding."""
        waveform = self._load_audio(audio_path).to(self.device)
        embedding = self.model.encode_batch(waveform).squeeze().float()
        return torch.nn.functional.normalize(embedding, p=2, dim=0).cpu()

    @torch.inference_mode()
    def verify(self, reference_audio: str | Path, incoming_audio: str | Path) -> dict[str, Any]:
        """Compare two recordings and return the requested verification result."""
        reference = self.embedding(reference_audio)
        incoming = self.embedding(incoming_audio)
        similarity = torch.nn.functional.cosine_similarity(reference, incoming, dim=0).item()
        return {
            "speaker_similarity": round(float(similarity), 4),
            "speaker_match": bool(similarity >= self.threshold),
        }


_default_verifier: SpeakerVerifier | None = None


def verify_speaker(
    reference_audio: str | Path,
    incoming_audio: str | Path,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, Any]:
    """Verify an incoming voice against a reference voice.

    The model is loaded once per process for the default threshold.
    """
    global _default_verifier
    if _default_verifier is None or _default_verifier.threshold != threshold:
        _default_verifier = SpeakerVerifier(threshold=threshold)
    return _default_verifier.verify(reference_audio, incoming_audio)