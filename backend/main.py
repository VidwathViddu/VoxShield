from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from tempfile import NamedTemporaryFile
import shutil

from risk_engine import calculate_risk
from speaker_verification.service import SpeakerVerifier

app = FastAPI(title="VoxShield API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REFERENCE_DIR = Path("data/reference_embeddings")
REFERENCE_USER = "demo_user"
REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

# Loaded once when the API starts. The ECAPA model is cached by SpeechBrain.
speaker_verifier = SpeakerVerifier()


async def _temporary_upload(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "audio.wav").suffix or ".wav"
    with NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        shutil.copyfileobj(upload.file, handle)
        return Path(handle.name)


@app.get("/")
def root():
    return {"message": "VoxShield API is running"}


@app.get("/speaker/status")
def speaker_status():
    return {
        "registered": (REFERENCE_DIR / f"{REFERENCE_USER}.pt").exists(),
        "user_id": REFERENCE_USER,
    }


@app.post("/speaker/enroll")
async def enroll_reference(file: UploadFile = File(...)):
    """Enroll the reference speaker by storing an ECAPA-TDNN embedding."""
    audio_path = await _temporary_upload(file)
    try:
        embedding = speaker_verifier.embedding(audio_path)
        output_path = REFERENCE_DIR / f"{REFERENCE_USER}.pt"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        import torch
        torch.save(embedding, output_path)

        return {
            "registered": True,
            "user_id": REFERENCE_USER,
            "embedding_dimensions": int(embedding.numel()),
        }
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not enroll the reference voice: {exc}",
        )
    finally:
        audio_path.unlink(missing_ok=True)


@app.post("/analyze")
async def analyze_voice(file: UploadFile = File(...)):
    """Analyze an incoming voice against the enrolled reference speaker."""
    import torch

    reference_path = REFERENCE_DIR / f"{REFERENCE_USER}.pt"
    if not reference_path.exists():
        raise HTTPException(
            status_code=400,
            detail="No reference voice is enrolled. Record the reference voice first.",
        )

    audio_path = await _temporary_upload(file)
    try:
        reference = torch.load(
            reference_path,
            map_location="cpu",
            weights_only=True,
        )
        incoming = speaker_verifier.embedding(audio_path)
        speaker_similarity = torch.nn.functional.cosine_similarity(
            reference, incoming, dim=0
        ).item()

        # TEMPORARY: replace this with Member 1's deepfake model output.
        spoof_probability = 0.10

        risk_result = calculate_risk(
            spoof_probability,
            speaker_similarity,
        )

        return {
            "filename": file.filename,
            "spoof_probability": spoof_probability,
            "speaker_similarity": round(float(speaker_similarity), 4),
            "speaker_match": bool(
                speaker_similarity >= speaker_verifier.threshold
            ),
            "authenticity_score": round((1 - spoof_probability) * 100),
            "risk_level": risk_result["risk_level"],
            "decision": risk_result["decision"],
        }
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not analyze the voice: {exc}",
        )
    finally:
        audio_path.unlink(missing_ok=True)
