# VoxShield — Phase 1 Prototype

## What works
- Live microphone recording using the browser MediaRecorder API
- Audio-file upload
- Playback of captured/uploaded audio
- Clean SIH-ready dashboard UI
- Placeholder AI-analysis pipeline for Phase 2

## Run
1. Install Node.js 18+.
2. In this folder run:
   npm install
   npm run dev
3. Open the local URL shown by Vite.
4. Allow microphone access to test recording.

## Next phase
Connect the recorded/uploaded audio to a Python/FastAPI endpoint and add the actual voice-deepfake model.


## Speaker verification integration

The prototype now has a two-stage voice flow:

1. Record/upload a **reference voice** and click **Enroll Reference Voice**.
2. Record/upload an **incoming voice** and click **Analyze Voice**.
3. FastAPI extracts an ECAPA-TDNN embedding for the enrolled voice and compares it with the incoming voice using cosine similarity.
4. The existing risk engine uses the real speaker similarity.
5. The deepfake/spoof probability is still a **temporary placeholder (0.10)** until Member 1's detector is integrated.

### Run locally

Terminal 1:
```powershell
cd backend
python -m uvicorn main:app --reload
```

Terminal 2:
```powershell
npm run dev
```

Open the Vite URL, normally `http://localhost:5173`.

The browser recorder converts its WebM capture to WAV before sending it to FastAPI, so the speaker-verification service can read the recording reliably.

The reference embedding is stored locally at:
`backend/data/reference_embeddings/demo_user.pt`

This is a Phase 1 prototype storage approach; a production deployment should use authenticated user identities and protected storage.
