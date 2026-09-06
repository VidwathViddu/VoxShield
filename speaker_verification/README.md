# Speaker verification — Member 2

This module establishes whether an incoming voice is likely to be the claimed
speaker. It uses the pretrained [SpeechBrain ECAPA-TDNN VoxCeleb model](https://speechbrain.readthedocs.io/en/stable/tutorials/advanced/pre-trained-models-and-fine-tuning-with-huggingface.html), not a model trained from scratch.

```
reference audio -> ECAPA-TDNN -> reference embedding
incoming audio  -> ECAPA-TDNN -> incoming embedding
                                      |
                         cosine similarity -> match decision
```

## 1. Create the environment

From the `VoxShield` folder, use your standard `python` command:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchaudio
pip install -r speaker_verification\requirements.txt
```

If `import torch` reports `DLL load failed` on Windows, install or repair the
current **Microsoft Visual C++ Redistributable x64**, restart the terminal, and
recreate the virtual environment. Check the setup before continuing:

```powershell
python -c "import torch, torchaudio, speechbrain; print(torch.__version__)"
```

The first verification downloads the pretrained model to
`pretrained_models/spkrec-ecapa-voxceleb`.

## 2. Prove the embedding model works

Record at least 10 seconds each of: `reference.wav` (you), `same.wav` (you
again), and `different.wav` (another person). Then run:

```powershell
python speaker_verification\evaluate_pairs.py reference.wav same.wav different.wav
```

The same-speaker score should generally be higher than the different-speaker
score. Do **not** treat one example as accuracy evidence; use your collected
test set to choose the operating threshold. The initial threshold is `0.65` and
is only a starting point.

## 3. Use the required Python function

```python
from speaker_verification import verify_speaker

result = verify_speaker("reference.wav", "incoming.wav")
# {"speaker_similarity": 0.84, "speaker_match": True}
```

The input may be WAV, MP3, FLAC, or another format supported by your installed
TorchAudio backend. Audio is converted to mono, 16 kHz before embedding.

## 4. Register a reference voice and verify an incoming call

Start the optional backend adapter:

```powershell
uvicorn speaker_verification.api:app --reload --port 8000
```

Register a user (the server stores only the derived embedding in
`data/reference_embeddings`):

```powershell
curl.exe -X POST -F "reference_audio=@reference.wav" http://127.0.0.1:8000/register/viddu
```

Verify a call-time recording:

```powershell
curl.exe -X POST -F "incoming_audio=@incoming.wav" http://127.0.0.1:8000/verify/viddu
```

Example result:

```json
{"speaker_similarity": 0.27, "speaker_match": false}
```

## 5. Test checklist

Collect labelled recordings and record every score/result for:

- same person, different phrases and sessions;
- different people;
- AI clone of the reference speaker;
- noisy recordings;
- different microphones or phone/network codecs.

Speaker verification and deepfake detection are separate decisions. An AI clone
may match the reference speaker embedding, so pair this module with the team's
deepfake detector before accepting a call as legitimate.
