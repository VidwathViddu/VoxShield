"""Optional FastAPI adapter for registration and call-time speaker verification."""

from __future__ import annotations

import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile

from .service import SpeakerVerifier

REFERENCE_DIR = Path("data/reference_embeddings")
verifier = SpeakerVerifier()
app = FastAPI(title="VoxShield Speaker Verification")


def _embedding_path(user_id: str) -> Path:
    if not user_id.replace("-", "").replace("_", "").isalnum():
        raise HTTPException(400, "user_id may contain only letters, numbers, hyphens, and underscores")
    return REFERENCE_DIR / f"{user_id}.pt"


async def _temporary_upload(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "audio.wav").suffix or ".wav"
    with NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        shutil.copyfileobj(upload.file, handle)
        return Path(handle.name)


@app.post("/register/{user_id}")
async def register_reference(user_id: str, reference_audio: UploadFile = File(...)):
    """Create or replace a user's stored reference speaker embedding."""
    audio_path = await _temporary_upload(reference_audio)
    try:
        embedding = verifier.embedding(audio_path)
        output_path = _embedding_path(user_id)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(embedding, output_path)
        return {"user_id": user_id, "registered": True, "embedding_dimensions": embedding.numel()}
    finally:
        audio_path.unlink(missing_ok=True)


@app.post("/verify/{user_id}")
async def verify_registered_speaker(user_id: str, incoming_audio: UploadFile = File(...)):
    """Compare an incoming recording with the saved reference for ``user_id``."""
    reference_path = _embedding_path(user_id)
    if not reference_path.exists():
        raise HTTPException(404, "No reference voice is registered for this user")

    audio_path = await _temporary_upload(incoming_audio)
    try:
        reference = torch.load(reference_path, map_location="cpu", weights_only=True)
        incoming = verifier.embedding(audio_path)
        similarity = torch.nn.functional.cosine_similarity(reference, incoming, dim=0).item()
        return {
            "speaker_similarity": round(float(similarity), 4),
            "speaker_match": bool(similarity >= verifier.threshold),
        }
    finally:
        audio_path.unlink(missing_ok=True)
