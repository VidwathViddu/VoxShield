import sys
from pathlib import Path

# Make sure local modules can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent))

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

ROOT_DIR = Path(__file__).resolve().parent

DATA_DIR = ROOT_DIR / "data" / "train_subset"

SAMPLE_RATE = 16000
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256

# All Mel spectrograms will have this many time frames
TARGET_FRAMES = 256

BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


print("================================")
print("VoxShield Deepfake Detector")
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

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


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

        # Load audio at 16 kHz
        audio, sample_rate = load_audio(audio_path)

        # Normalize audio
        audio = preprocess_audio(audio)

        # Extract Mel spectrogram
        mel = extract_mel_spectrogram(
            audio,
            sample_rate,
            n_mels=N_MELS,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )

        # ====================================================
        # Make every Mel spectrogram the same size
        # ====================================================

        if mel.shape[1] < TARGET_FRAMES:

            # Short audio -> pad
            pad_width = TARGET_FRAMES - mel.shape[1]

            mel = np.pad(
                mel,
                ((0, 0), (0, pad_width)),
                mode="constant",
                constant_values=mel.min()
            )

        else:

            # Long audio -> crop
            mel = mel[:, :TARGET_FRAMES]

        # ====================================================
        # Convert to tensor
        # ====================================================

        mel = torch.tensor(
            mel,
            dtype=torch.float32
        )

        # ====================================================
        # Normalize spectrogram
        # ====================================================

        mel = (
            mel - mel.mean()
        ) / (
            mel.std() + 1e-6
        )

        # ====================================================
        # CNN expects:
        # [channels, height, width]
        # ====================================================

        mel = mel.unsqueeze(0)

        return (
            mel,
            torch.tensor(
                label,
                dtype=torch.long
            )
        )


# ============================================================
# Collect files
# ============================================================

bonafide_files = sorted(
    DATA_DIR.glob("bonafide_*.flac")
)

spoof_files = sorted(
    DATA_DIR.glob("spoof_*.flac")
)


print("\nDataset:")
print("Bonafide:", len(bonafide_files))
print("Spoof:", len(spoof_files))


files = []

# Label 0 = Bonafide
for path in bonafide_files:
    files.append(
        (path, 0)
    )

# Label 1 = Spoof
for path in spoof_files:
    files.append(
        (path, 1)
    )


# ============================================================
# Shuffle before splitting
# ============================================================

random.shuffle(files)


# ============================================================
# 80% Training / 20% Validation
# ============================================================

split_index = int(
    len(files) * 0.8
)

train_files = files[:split_index]
val_files = files[split_index:]


print(
    "Training samples:",
    len(train_files)
)

print(
    "Validation samples:",
    len(val_files)
)


# ============================================================
# DataLoaders
# ============================================================

train_dataset = VoiceDataset(
    train_files
)

val_dataset = VoiceDataset(
    val_files
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
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

            # -----------------------------------------------
            # Convolution Block 1
            # -----------------------------------------------

            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(2),


            # -----------------------------------------------
            # Convolution Block 2
            # -----------------------------------------------

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.MaxPool2d(2),


            # -----------------------------------------------
            # Convolution Block 3
            # -----------------------------------------------

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(),


            # -----------------------------------------------
            # Global pooling
            # -----------------------------------------------

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
# Initialize Model
# ============================================================

model = DeepfakeCNN().to(DEVICE)


criterion = nn.CrossEntropyLoss()


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# Model Saving
# ============================================================

best_val_accuracy = 0.0

MODEL_DIR = ROOT_DIR / "models"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_PATH = (
    MODEL_DIR /
    "voxshield_deepfake_cnn.pt"
)


# ============================================================
# Training
# ============================================================

for epoch in range(EPOCHS):

    # ========================================================
    # Training
    # ========================================================

    model.train()

    train_correct = 0
    train_total = 0
    train_loss = 0.0


    for mel, labels in train_loader:

        mel = mel.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )


        # Clear gradients
        optimizer.zero_grad()


        # Forward pass
        outputs = model(mel)


        # Calculate loss
        loss = criterion(
            outputs,
            labels
        )


        # Backpropagation
        loss.backward()


        # Update weights
        optimizer.step()


        # Track loss
        train_loss += loss.item()


        # Predictions
        predictions = outputs.argmax(
            dim=1
        )


        train_correct += (
            predictions == labels
        ).sum().item()


        train_total += labels.size(0)


    train_accuracy = (
        train_correct /
        train_total
    )


    # ========================================================
    # Validation
    # ========================================================

    model.eval()

    val_correct = 0
    val_total = 0
    val_loss = 0.0


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


            # Forward pass
            outputs = model(mel)


            # Calculate validation loss
            loss = criterion(
                outputs,
                labels
            )


            val_loss += loss.item()


            # Predictions
            predictions = outputs.argmax(
                dim=1
            )


            val_correct += (
                predictions == labels
            ).sum().item()


            val_total += labels.size(0)


    val_accuracy = (
        val_correct /
        val_total
    )


    # ========================================================
    # Epoch Results
    # ========================================================

    print(
        f"\nEpoch {epoch + 1}/{EPOCHS}"
    )


    print(
        f"Train Loss: "
        f"{train_loss / len(train_loader):.4f}"
    )


    print(
        f"Train Accuracy: "
        f"{train_accuracy * 100:.2f}%"
    )


    print(
        f"Validation Loss: "
        f"{val_loss / len(val_loader):.4f}"
    )


    print(
        f"Validation Accuracy: "
        f"{val_accuracy * 100:.2f}%"
    )


    # ========================================================
    # Save Best Model
    # ========================================================

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy


        torch.save(
            model.state_dict(),
            MODEL_PATH
        )


        print(
            "✓ Best model saved"
        )


# ============================================================
# Training Complete
# ============================================================

print("\n================================")
print("Training complete")
print("================================")


print(
    f"Best validation accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)


print(
    f"Model saved to: "
    f"{MODEL_PATH}"
)