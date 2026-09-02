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
