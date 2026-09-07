from preprocessing.audio import load_audio, preprocess_audio


audio_path = "deepfake_detector/data/test_audio.wav"

audio, sample_rate = load_audio(audio_path)

print("================================")
print("VoxShield Audio Test")
print("================================")

print("Sample rate:", sample_rate)
print("Number of samples:", len(audio))
print("Duration:", len(audio) / sample_rate, "seconds")

audio = preprocess_audio(audio)

print("Maximum amplitude:", audio.max())
print("Minimum amplitude:", audio.min())
print("Audio preprocessing successful!")