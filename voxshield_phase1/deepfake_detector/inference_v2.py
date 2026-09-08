import sys
from pathlib import Path

# Make local modules importable
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
import torch.nn as nn

from preprocessing.audio import load_audio, preprocess_audio
from preprocessing.features import extract_mel_spectrogram


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = (
    ROOT_DIR
    / "models"
    / "voxshield_deepfake_cnn_v2.pt"
)

SAMPLE_RATE = 16000
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256
TARGET_FRAMES = 256

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CNN Model
# ============================================================

class DeepfakeCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(128, 64),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(64, 2)
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


# ============================================================
# Load Model
# ============================================================

model = DeepfakeCNN().to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )
)

model.eval()


# ============================================================
# Prediction Function
# ============================================================

@torch.no_grad()
def predict_deepfake(audio_path):

    # --------------------------------------------
    # Load audio
    # --------------------------------------------

    audio, sample_rate = load_audio(
        audio_path
    )

    # --------------------------------------------
    # Preprocess
    # --------------------------------------------

    audio = preprocess_audio(
        audio
    )

    # --------------------------------------------
    # Extract Mel spectrogram
    # --------------------------------------------

    mel = extract_mel_spectrogram(
        audio,
        sample_rate,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    # --------------------------------------------
    # Fix spectrogram length
    # --------------------------------------------

    if mel.shape[1] < TARGET_FRAMES:

        pad_width = TARGET_FRAMES - mel.shape[1]

        mel = np.pad(
            mel,
            ((0, 0), (0, pad_width)),
            mode="constant",
            constant_values=mel.min()
        )

    else:

        mel = mel[:, :TARGET_FRAMES]

    # --------------------------------------------
    # Convert to tensor
    # --------------------------------------------

    mel = torch.tensor(
        mel,
        dtype=torch.float32
    )

    # --------------------------------------------
    # Normalize
    # --------------------------------------------

    mel = (
        mel - mel.mean()
    ) / (
        mel.std() + 1e-6
    )

    # --------------------------------------------
    # Add batch + channel dimensions
    #
    # [128, 256]
    #      ↓
    # [1, 128, 256]
    #      ↓
    # [1, 1, 128, 256]
    # --------------------------------------------

    mel = mel.unsqueeze(0).unsqueeze(0)

    mel = mel.to(DEVICE)

    # --------------------------------------------
    # CNN prediction
    # --------------------------------------------

    outputs = model(mel)

    probabilities = torch.softmax(
        outputs,
        dim=1
    )

    # Label:
    # 0 = Bonafide
    # 1 = Spoof

    bonafide_probability = (
        probabilities[0][0].item()
    )

    spoof_probability = (
        probabilities[0][1].item()
    )

    prediction = (
        "SPOOF"
        if spoof_probability >= bonafide_probability
        else "BONAFIDE"
    )

    return {
        "prediction": prediction,
        "bonafide_probability": round(
            bonafide_probability,
            4
        ),
        "spoof_probability": round(
            spoof_probability,
            4
        )
    }


# ============================================================
# Test from command line
# ============================================================

if __name__ == "__main__":

    print("================================")
    print("VoxShield Deepfake Inference")
    print("================================")

    print("Device:", DEVICE)

    print("Model:", MODEL_PATH)

    if not MODEL_PATH.exists():

        print("\nERROR: Model file not found.")

        sys.exit(1)

    print("\n✓ Model loaded successfully")

    print(
        "\nUsage:"
    )

    print(
        "python inference.py <audio_file>"
    )

    if len(sys.argv) < 2:

        print(
            "\nNo audio file supplied."
        )

        sys.exit(0)

    audio_path = Path(
        sys.argv[1]
    )

    if not audio_path.exists():

        print(
            "\nERROR: Audio file not found:"
        )

        print(audio_path)

        sys.exit(1)

    print(
        "\nAnalyzing:",
        audio_path.name
    )

    result = predict_deepfake(
        audio_path
    )

    print("\n================================")
    print("Prediction")
    print("================================")

    print(
        "Result:",
        result["prediction"]
    )

    print(
        "Bonafide Probability:",
        f"{result['bonafide_probability'] * 100:.2f}%"
    )

    print(
        "Spoof Probability:",
        f"{result['spoof_probability'] * 100:.2f}%"
    )

    print("\n================================")
