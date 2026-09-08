from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from tempfile import NamedTemporaryFile
import shutil
import sys

import torch

from backend.risk_engine import calculate_risk
from speaker_verification.service import SpeakerVerifier


# ============================================================
# Make deepfake detector importable
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DEEPFAKE_DIR = PROJECT_DIR / "deepfake_detector"

sys.path.insert(0, str(DEEPFAKE_DIR))

from aasist_detector import AASISTDetector


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="VoxShield API"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://vox-shield-ebon.vercel.app",
],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Speaker Verification
# ============================================================

REFERENCE_DIR = (
    BASE_DIR
    / "data"
    / "reference_embeddings"
)

REFERENCE_USER = "demo_user"

REFERENCE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# Loaded once when API starts
speaker_verifier = SpeakerVerifier()

# AASIST anti-spoofing detector
aasist_detector = AASISTDetector()

# ============================================================
# Temporary Upload Helper
# ============================================================

async def _temporary_upload(
    upload: UploadFile
) -> Path:

    suffix = (
        Path(
            upload.filename or "audio.wav"
        ).suffix
        or ".wav"
    )

    with NamedTemporaryFile(
        suffix=suffix,
        delete=False
    ) as handle:

        shutil.copyfileobj(
            upload.file,
            handle
        )

        return Path(handle.name)


# ============================================================
# Root
# ============================================================

@app.get("/")
def root():

    return {
        "message": "VoxShield API is running"
    }


# ============================================================
# Speaker Status
# ============================================================

@app.get("/speaker/status")
def speaker_status():
    """
    Return the current speaker-profile status.

    The normal .pt file contains the ECAPA centroid.
    The _multi.pt file contains the individual reference embeddings.
    """

    profile_path = (
        REFERENCE_DIR
        / f"{REFERENCE_USER}.pt"
    )

    multi_profile_path = (
        REFERENCE_DIR
        / f"{REFERENCE_USER}_multi.pt"
    )

    reference_count = 0

    if multi_profile_path.exists():
        try:
            data = torch.load(
                multi_profile_path,
                map_location="cpu",
                weights_only=True,
            )

            reference_count = int(
                data.get("count", 0)
            )

        except Exception:
            reference_count = 0

    return {
        "registered": profile_path.exists(),
        "user_id": REFERENCE_USER,
        "reference_count": reference_count,
        "profile_type": (
            "multi_reference"
            if reference_count > 1
            else "single_reference"
        ),
    }


# ============================================================
# Reset Speaker Profile
# ============================================================

@app.post("/speaker/reset")
def reset_speaker_profile():
    """
    Reset the active speaker profile.

    This only removes the enrolled speaker embeddings.
    Existing analysis results and history are NOT affected.
    """

    profile_path = (
        REFERENCE_DIR
        / f"{REFERENCE_USER}.pt"
    )

    multi_profile_path = (
        REFERENCE_DIR
        / f"{REFERENCE_USER}_multi.pt"
    )

    try:
        profile_path.unlink(missing_ok=True)
        multi_profile_path.unlink(missing_ok=True)

        return {
            "registered": False,
            "reference_count": 0,
            "message": "Speaker profile reset successfully.",
        }

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not reset speaker profile: {exc}",
        )

    
# ============================================================
# Speaker Enrollment
# ============================================================

@app.post("/speaker/enroll-experiment")
async def enroll_reference_experiment(file: UploadFile = File(...)):
    """
    Experimental multi-reference ECAPA enrollment.

    Each uploaded recording contributes one embedding to the
    speaker profile. The existing /speaker/enroll endpoint is untouched.
    """
    profile_path = REFERENCE_DIR / f"{REFERENCE_USER}_multi.pt"

    audio_path = await _temporary_upload(file)

    try:
        embedding = speaker_verifier.embedding(audio_path)

        if profile_path.exists():
            data = torch.load(
                profile_path,
                map_location="cpu",
                weights_only=True,
            )
            embeddings = data["embeddings"]
        else:
            embeddings = []

        embeddings.append(embedding.cpu())

        profile = speaker_verifier.build_profile(
            reference_audio_paths=[]
        ) if False else None

        torch.save(
            {
                "embeddings": embeddings,
                "count": len(embeddings),
            },
            profile_path,
        )

        return {
            "registered": True,
            "experimental": True,
            "reference_count": len(embeddings),
            "message": f"Reference {len(embeddings)} added successfully.",
        }

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not add experimental reference voice: {exc}",
        )

    finally:
        audio_path.unlink(missing_ok=True)

@app.post("/speaker/enroll")
async def enroll_reference(
    file: UploadFile = File(...)
):
    """
    Enroll a reference speaker.

    Each enrollment contributes one ECAPA embedding to a
    small multi-reference speaker profile.

    The centroid is also stored in the existing .pt file so
    the rest of VoxShield can continue using the same reference path.
    """

    audio_path = await _temporary_upload(file)

    try:

        # --------------------------------------------------------
        # Extract embedding from this reference recording
        # --------------------------------------------------------

        embedding = speaker_verifier.embedding(
            audio_path
        )

        multi_profile_path = (
            REFERENCE_DIR
            / f"{REFERENCE_USER}_multi.pt"
        )

        # --------------------------------------------------------
        # Load existing reference embeddings
        # --------------------------------------------------------

        if multi_profile_path.exists():

            data = torch.load(
                multi_profile_path,
                map_location="cpu",
                weights_only=True,
            )

            embeddings = data.get(
                "embeddings",
                [],
            )

        else:

            embeddings = []

        # --------------------------------------------------------
        # Add new reference
        # --------------------------------------------------------

        embeddings.append(
            embedding.cpu()
        )

        # Keep the profile small and controlled.
        # Most recent 5 reference recordings are retained.
        embeddings = embeddings[-5:]

        # --------------------------------------------------------
        # Build centroid
        # --------------------------------------------------------

        profile = speaker_verifier.build_profile(
            embeddings
        )

        # --------------------------------------------------------
        # Save multi-reference profile
        # --------------------------------------------------------

        torch.save(
            {
                "embeddings": embeddings,
                "count": len(embeddings),
            },
            multi_profile_path,
        )

        # --------------------------------------------------------
        # IMPORTANT:
        # Keep the original .pt path used by VoxShield.
        # It now contains the multi-reference centroid.
        # --------------------------------------------------------

        output_path = (
            REFERENCE_DIR
            / f"{REFERENCE_USER}.pt"
        )

        torch.save(
            profile,
            output_path,
        )

        return {
            "registered": True,
            "user_id": REFERENCE_USER,
            "embedding_dimensions": int(
                profile.numel()
            ),
            "reference_count": len(
                embeddings
            ),
            "message": (
                f"Reference profile updated "
                f"using {len(embeddings)} recording(s)."
            ),
        }

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail=(
                "Could not enroll the "
                f"reference voice: {exc}"
            ),
        )

    finally:

        audio_path.unlink(
            missing_ok=True
        )

    """
    Enroll the reference speaker by
    storing an ECAPA-TDNN embedding.
    """

    audio_path = (
        await _temporary_upload(file)
    )

    try:

        embedding = (
            speaker_verifier.embedding(
                audio_path
            )
        )

        output_path = (
            REFERENCE_DIR
            / f"{REFERENCE_USER}.pt"
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        torch.save(
            embedding,
            output_path
        )

        return {
            "registered": True,
            "user_id": REFERENCE_USER,
            "embedding_dimensions": int(
                embedding.numel()
            ),
        }

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail=(
                "Could not enroll the "
                f"reference voice: {exc}"
            ),
        )

    finally:

        audio_path.unlink(
            missing_ok=True
        )


# ============================================================
# Voice Analysis
# ============================================================

@app.post("/analyze")
async def analyze_voice(
    file: UploadFile = File(...)
):

    """
    Analyze incoming audio using:

    1. AASIST anti-spoofing detector
    2. ECAPA-TDNN speaker verification
    3. VoxShield risk engine
    """

    # --------------------------------------------------------
    # Check reference speaker
    # --------------------------------------------------------

    reference_path = (
        REFERENCE_DIR
        / f"{REFERENCE_USER}.pt"
    )

    if not reference_path.exists():

        raise HTTPException(
            status_code=400,
            detail=(
                "No reference voice is enrolled. "
                "Record the reference voice first."
            ),
        )

    # --------------------------------------------------------
    # Save uploaded audio temporarily
    # --------------------------------------------------------

    audio_path = await _temporary_upload(file)

    try:

        # ====================================================
        # STEP 1 — AASIST Anti-Spoofing Detection
        # ====================================================

        deepfake_result = aasist_detector.predict(
            audio_path
        )

        spoof_probability = deepfake_result[
            "spoof_probability"
        ]

        # --------------------------------------------------------
        # STEP 2 — Speaker Verification
        # --------------------------------------------------------

        reference = torch.load(
            reference_path,
            map_location="cpu",
            weights_only=True,
        )

        speaker_result = speaker_verifier.compare_profile(
            reference,
            audio_path,
        )

        speaker_similarity = speaker_result[
            "speaker_similarity"
        ]

        speaker_match = speaker_result[
            "speaker_match"
        ]
        # ====================================================
        # STEP 3 — Risk Engine
        # ====================================================

        risk_result = calculate_risk(
            spoof_probability,
            speaker_similarity
        )

        # ====================================================
        # STEP 4 — Authenticity Score
        # ====================================================

        authenticity_score = round(
            (1 - spoof_probability) * 100
        )

        # ====================================================
        # STEP 5 — Return Combined Result
        # ====================================================

        return {

            "filename": file.filename,

            # AASIST anti-spoofing
            "prediction": deepfake_result[
                "label"
            ],

            "spoof_probability": (
                spoof_probability
            ),

            "bonafide_probability": (
                deepfake_result[
                    "bonafide_probability"
                ]
            ),

            # Speaker verification
            "speaker_similarity": round(
                float(speaker_similarity),
                4
            ),

            "speaker_match": speaker_match,

            # Risk assessment
            "authenticity_score": authenticity_score,

            "risk_level": (
                risk_result[
                    "risk_level"
                ]
            ),

            "decision": (
                risk_result[
                    "decision"
                ]
            ),
        }

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail=(
                "Could not analyze "
                f"the voice: {exc}"
            ),
        )

    finally:

        audio_path.unlink(
            missing_ok=True
        )
@app.post("/analyze-experiment")
async def analyze_voice_experiment(file: UploadFile = File(...)):
    """
    Experimental analysis using the multi-reference ECAPA profile.
    The existing /analyze endpoint is intentionally untouched.
    """
    import torch

    profile_path = REFERENCE_DIR / f"{REFERENCE_USER}_multi.pt"

    if not profile_path.exists():
        raise HTTPException(
            status_code=400,
            detail="No experimental speaker profile is enrolled."
        )

    audio_path = await _temporary_upload(file)

    try:
        data = torch.load(
            profile_path,
            map_location="cpu",
            weights_only=True
        )

        embeddings = data["embeddings"]

        if not embeddings:
            raise ValueError("Experimental speaker profile is empty.")

        # Create centroid from all reference embeddings
        profile = torch.stack(embeddings).mean(dim=0)
        profile = torch.nn.functional.normalize(
            profile.float(),
            p=2,
            dim=0
        )

        incoming = speaker_verifier.embedding(audio_path)

        similarity = torch.nn.functional.cosine_similarity(
            profile,
            incoming,
            dim=0
        ).item()

        speaker_match = bool(
            similarity >= speaker_verifier.threshold
        )

        return {
            "experimental": True,
            "reference_count": len(embeddings),
            "speaker_similarity": round(float(similarity), 4),
            "speaker_similarity_percent": round(float(similarity) * 100, 2),
            "speaker_match": speaker_match,
            "threshold": speaker_verifier.threshold,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not analyze experimental voice: {exc}"
        )

    finally:
        audio_path.unlink(missing_ok=True)