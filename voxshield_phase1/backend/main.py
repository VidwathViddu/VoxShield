from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="VoxShield API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "VoxShield API is running"}


@app.post("/analyze")
async def analyze_voice(file: UploadFile = File(...)):
    return {
        "filename": file.filename,
        "spoof_probability": 0.10,
        "speaker_similarity": 0.92,
        "authenticity_score": 90,
        "risk_level": "LOW",
        "decision": "VOICE_APPEARS_AUTHENTIC"
    }