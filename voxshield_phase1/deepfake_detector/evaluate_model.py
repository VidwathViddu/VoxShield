import sys
from pathlib import Path

# Make sure local modules can be imported
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from preprocessing.audio import load_audio, preprocess_audio
from preprocessing.features import extract_mel_spectrogram


# ============================================================
# Configuration
# ============================================================

DATA_DIR = ROOT_DIR / "data" / "train_subset"

MODEL_PATH = (
    ROOT_DIR
    / "models"
    / "voxshield_deepfake_cnn.pt"
)

SAMPLE_RATE = 16000
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256
TARGET_FRAMES = 256

BATCH_SIZE = 32
SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Header
# ============================================================

print("================================")
print("VoxShield Model Evaluation")
print("================================")

print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# Dataset
# ============================================================

class VoiceDataset(Dataset):

    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        audio_path, label = self.files[index]

        audio, sample_rate = load_audio(audio_path)

        audio = preprocess_audio(audio)

        mel = extract_mel_spectrogram(
            audio,
            sample_rate,
            n_mels=N_MELS,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )

        # ====================================================
        # Make all spectrograms the same size
        # ====================================================

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

        # ====================================================
        # Convert to tensor
        # ====================================================

        mel = torch.tensor(
            mel,
            dtype=torch.float32
        )

        # Normalize spectrogram
        mel = (
            mel - mel.mean()
        ) / (
            mel.std() + 1e-6
        )

        # Add channel dimension
        mel = mel.unsqueeze(0)

        return (
            mel,
            torch.tensor(
                label,
                dtype=torch.long
            )
        )


# ============================================================
# Collect Dataset
# ============================================================

bonafide_files = sorted(
    DATA_DIR.glob("bonafide_*.flac")
)

spoof_files = sorted(
    DATA_DIR.glob("spoof_*.flac")
)


files = []

# Label 0 = Bonafide
for path in bonafide_files:
    files.append((path, 0))

# Label 1 = Spoof
for path in spoof_files:
    files.append((path, 1))


# ============================================================
# IMPORTANT:
# Use the EXACT same random split as training
# ============================================================

random.shuffle(files)

split_index = int(
    len(files) * 0.8
)

train_files = files[:split_index]
val_files = files[split_index:]


print("\nDataset:")
print("Total samples:", len(files))
print("Validation samples:", len(val_files))


# ============================================================
# Validation DataLoader
# ============================================================

val_dataset = VoiceDataset(
    val_files
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
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

            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
        )


        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                128,
                64
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                64,
                2
            )
        )


    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


# ============================================================
# Load Saved Model
# ============================================================

print("\nLoading model...")

model = DeepfakeCNN().to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )
)

model.eval()

print("✓ Model loaded successfully")


# ============================================================
# Evaluation
# ============================================================

correct = 0
total = 0

true_positives = 0
true_negatives = 0
false_positives = 0
false_negatives = 0


with torch.no_grad():

    for mel, labels in val_loader:

        mel = mel.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        outputs = model(mel)

        predictions = outputs.argmax(
            dim=1
        )


        # -----------------------------------------------
        # Accuracy
        # -----------------------------------------------

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)


        # -----------------------------------------------
        # Confusion Matrix
        #
        # 0 = Bonafide
        # 1 = Spoof
        # -----------------------------------------------

        true_positives += (
            ((predictions == 1) & (labels == 1))
        ).sum().item()

        true_negatives += (
            ((predictions == 0) & (labels == 0))
        ).sum().item()

        false_positives += (
            ((predictions == 1) & (labels == 0))
        ).sum().item()

        false_negatives += (
            ((predictions == 0) & (labels == 1))
        ).sum().item()


# ============================================================
# Metrics
# ============================================================

accuracy = (
    correct / total
    if total > 0
    else 0
)


precision = (
    true_positives /
    (true_positives + false_positives)
    if (true_positives + false_positives) > 0
    else 0
)


recall = (
    true_positives /
    (true_positives + false_negatives)
    if (true_positives + false_negatives) > 0
    else 0
)


f1_score = (
    2 * precision * recall /
    (precision + recall)
    if (precision + recall) > 0
    else 0
)


# ============================================================
# Results
# ============================================================

print("\n================================")
print("Evaluation Results")
print("================================")

print(
    f"Accuracy:  {accuracy * 100:.2f}%"
)

print(
    f"Precision: {precision * 100:.2f}%"
)

print(
    f"Recall:    {recall * 100:.2f}%"
)

print(
    f"F1 Score:  {f1_score * 100:.2f}%"
)


# ============================================================
# Confusion Matrix
# ============================================================

print("\nConfusion Matrix:")
print("--------------------------------")

print(
    "                 Predicted"
)

print(
    "               Bonafide  Spoof"
)

print(
    f"Actual Bonafide   "
    f"{true_negatives:4d}    "
    f"{false_positives:4d}"
)

print(
    f"Actual Spoof      "
    f"{false_negatives:4d}    "
    f"{true_positives:4d}"
)


# ============================================================
# Detailed Counts
# ============================================================

print("\nDetailed Results:")
print("--------------------------------")

print(
    "True Negatives:",
    true_negatives
)

print(
    "False Positives:",
    false_positives
)

print(
    "False Negatives:",
    false_negatives
)

print(
    "True Positives:",
    true_positives
)


print("\n================================")
print("Evaluation complete")
print("================================")