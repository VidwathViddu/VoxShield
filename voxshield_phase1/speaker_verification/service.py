"""
ECAPA-TDNN speaker embedding extraction and speaker verification.

This module verifies speaker identity.
It does NOT determine whether audio is human or synthetically generated.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import soundfile as sf
import torch
import torchaudio
from speechbrain.inference.speaker import SpeakerRecognition
from speechbrain.utils.fetching import LocalStrategy


MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"

SAMPLE_RATE = 16_000

DEFAULT_THRESHOLD = 0.65


class SpeakerVerifier:
    """
    Pretrained ECAPA-TDNN speaker embedding extractor and verifier.

    The threshold is configurable and should ultimately be calibrated
    using genuine and impostor recordings from the deployment environment.
    """

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        model_dir: str | Path = "pretrained_models/spkrec-ecapa-voxceleb",
        device: str | None = None,
    ) -> None:

        if not -1.0 <= threshold <= 1.0:
            raise ValueError(
                "threshold must be between -1.0 and 1.0"
            )

        self.threshold = threshold

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = SpeakerRecognition.from_hparams(
            source=MODEL_ID,
            savedir=str(model_dir),
            local_strategy=LocalStrategy.COPY,
            run_opts={"device": self.device},
        )

    # ============================================================
    # AUDIO
    # ============================================================

    @staticmethod
    def _load_audio(
        audio_path: str | Path,
    ) -> torch.Tensor:
        """
        Load an audio file as a mono 16 kHz waveform.
        """

        data, sample_rate = sf.read(
            str(audio_path),
            dtype="float32",
            always_2d=True,
        )

        waveform = torch.from_numpy(data.T)

        if waveform.numel() == 0:
            raise ValueError(
                f"Audio file is empty: {audio_path}"
            )

        # Convert stereo/multi-channel audio to mono
        if waveform.size(0) > 1:
            waveform = waveform.mean(
                dim=0,
                keepdim=True,
            )

        # Resample to ECAPA's expected sample rate
        if sample_rate != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(
                waveform,
                sample_rate,
                SAMPLE_RATE,
            )

        return waveform

    # ============================================================
    # EMBEDDING
    # ============================================================

    @torch.inference_mode()
    def embedding(
        self,
        audio_path: str | Path,
    ) -> torch.Tensor:
        """
        Return a unit-normalized ECAPA speaker embedding.
        """

        waveform = self._load_audio(audio_path).to(
            self.device
        )

        embedding = (
            self.model
            .encode_batch(waveform)
            .squeeze()
            .float()
        )

        return torch.nn.functional.normalize(
            embedding,
            p=2,
            dim=0,
        ).cpu()

    # ============================================================
    # MULTI-REFERENCE PROFILE
    # ============================================================

    @torch.inference_mode()
    def build_profile(
        self,
        reference_embeddings: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Build a normalized speaker profile from multiple
        precomputed reference embeddings.
        """

        if not reference_embeddings:
            raise ValueError(
                "At least one reference embedding is required."
            )

        embeddings = [
            torch.nn.functional.normalize(
                embedding.float(),
                p=2,
                dim=0,
            )
            for embedding in reference_embeddings
        ]

        profile = torch.stack(
            embeddings
        ).mean(dim=0)

        return torch.nn.functional.normalize(
            profile,
            p=2,
            dim=0,
        )

    # ============================================================
    # PROFILE COMPARISON
    # ============================================================

    @torch.inference_mode()
    def compare_profile(
        self,
        profile: torch.Tensor,
        incoming_audio: str | Path,
    ) -> dict[str, Any]:
        """
        Compare an incoming recording against a speaker profile.
        """

        incoming = self.embedding(
            incoming_audio
        )

        profile = torch.nn.functional.normalize(
            profile.float(),
            p=2,
            dim=0,
        )

        similarity = (
            torch.nn.functional.cosine_similarity(
                profile,
                incoming,
                dim=0,
            ).item()
        )

        return {
            "speaker_similarity": round(
                float(similarity),
                4,
            ),
            "speaker_match": bool(
                similarity >= self.threshold
            ),
        }

    # ============================================================
    # SINGLE REFERENCE VERIFICATION
    # ============================================================

    @torch.inference_mode()
    def verify(
        self,
        reference_audio: str | Path,
        incoming_audio: str | Path,
    ) -> dict[str, Any]:
        """
        Compare two recordings directly.
        """

        reference = self.embedding(
            reference_audio
        )

        incoming = self.embedding(
            incoming_audio
        )

        similarity = (
            torch.nn.functional.cosine_similarity(
                reference,
                incoming,
                dim=0,
            ).item()
        )

        return {
            "speaker_similarity": round(
                float(similarity),
                4,
            ),
            "speaker_match": bool(
                similarity >= self.threshold
            ),
        }


# ============================================================
# DEFAULT VERIFIER
# ============================================================

_default_verifier: SpeakerVerifier | None = None


def verify_speaker(
    reference_audio: str | Path,
    incoming_audio: str | Path,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, Any]:
    """
    Verify an incoming voice against a single reference voice.
    """

    global _default_verifier

    if (
        _default_verifier is None
        or _default_verifier.threshold != threshold
    ):
        _default_verifier = SpeakerVerifier(
            threshold=threshold
        )

    return _default_verifier.verify(
        reference_audio,
        incoming_audio,
    )