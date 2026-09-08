import os
import sys
import torch
import librosa
import numpy as np

# Add official AASIST repository to Python path
AASIST_DIR = os.path.join(os.path.dirname(__file__), "aasist")
sys.path.insert(0, AASIST_DIR)

from models.AASIST import Model


class AASISTDetector:

    TARGET_SR = 16000
    TARGET_LENGTH = 64600

    def __init__(self):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model_config = {
            "architecture": "AASIST",
            "nb_samp": 64600,
            "first_conv": 128,
            "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
            "gat_dims": [64, 32],
            "pool_ratios": [0.5, 0.7, 0.5, 0.5],
            "temperatures": [2.0, 2.0, 100.0, 100.0],
        }

        self.model = Model(self.model_config)

        model_path = os.path.join(
            AASIST_DIR,
            "models",
            "weights",
            "AASIST.pth"
        )

        checkpoint = torch.load(
            model_path,
            map_location=self.device
        )

        self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()

        print(f"AASIST loaded on {self.device}")

    def _load_audio(self, audio_path):

        audio, _ = librosa.load(
            audio_path,
            sr=self.TARGET_SR,
            mono=True
        )

        audio = audio.astype(np.float32)

        # Repeat short audio until it reaches the required length
        if len(audio) < self.TARGET_LENGTH:

            repeat_count = int(
                np.ceil(self.TARGET_LENGTH / len(audio))
            )

            audio = np.tile(audio, repeat_count)

        # Use exactly 64600 samples
        audio = audio[:self.TARGET_LENGTH]

        audio = torch.tensor(audio).unsqueeze(0)

        return audio.to(self.device)

    def predict(self, audio_path):

        audio = self._load_audio(audio_path)

        with torch.no_grad():

            _, output = self.model(audio)

            probabilities = torch.softmax(output, dim=1)

        # Verified using our genuine and AI test recordings:
        # Class 0 = BONAFIDE
        # Class 1 = SPOOF

        bonafide_probability = probabilities[0][0].item()
        spoof_probability = probabilities[0][1].item()

        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()

        if predicted_class == 1:
            label = "SPOOF"
        else:
            label = "BONAFIDE"

        return {
            "label": label,
            "spoof_probability": spoof_probability,
            "bonafide_probability": bonafide_probability,
            "predicted_class": predicted_class
        }


if __name__ == "__main__":

    detector = AASISTDetector()

    test_file = r"C:\Users\viddu\Downloads\ai1.wav"

    result = detector.predict(test_file)

    print("\nAASIST RESULT")
    print("-------------------------")
    print("Label:", result["label"])
    print(
        "Spoof Probability:",
        f"{result['spoof_probability'] * 100:.2f}%"
    )
    print(
        "Bonafide Probability:",
        f"{result['bonafide_probability'] * 100:.2f}%"
    )