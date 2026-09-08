from pathlib import Path
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

REPO_ID = "Bisher/ASVspoof_2019_LA"
PARQUET_FILE = "data/train-00000-of-00001.parquet"

OUTPUT_DIR = Path("deepfake_detector/data/train_subset")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_PER_CLASS = 2500

print("Loading cached dataset metadata...")

parquet_path = hf_hub_download(
    repo_id=REPO_ID,
    filename=PARQUET_FILE,
    repo_type="dataset"
)

table = pq.read_table(parquet_path)

rows = table.to_pylist()

bonafide_count = 0
spoof_count = 0

print("Extracting balanced subset...")

for row in rows:

    label = row["key"]
    audio = row["audio"]

    if audio is None or audio.get("bytes") is None:
        continue

    audio_data = audio["bytes"]

    if label == 0 and bonafide_count < TARGET_PER_CLASS:

        output_path = (
            OUTPUT_DIR /
            f"bonafide_{bonafide_count:05d}.flac"
        )

        output_path.write_bytes(audio_data)
        bonafide_count += 1

    elif label == 1 and spoof_count < TARGET_PER_CLASS:

        output_path = (
            OUTPUT_DIR /
            f"spoof_{spoof_count:05d}.flac"
        )

        output_path.write_bytes(audio_data)
        spoof_count += 1

    total = bonafide_count + spoof_count

    if total > 0 and total % 100 == 0:
        print(
            f"Bonafide: {bonafide_count}/{TARGET_PER_CLASS} | "
            f"Spoof: {spoof_count}/{TARGET_PER_CLASS}"
        )

    if (
        bonafide_count >= TARGET_PER_CLASS
        and spoof_count >= TARGET_PER_CLASS
    ):
        break

print("\n================================")
print("Subset extraction complete")
print("================================")
print(f"Bonafide: {bonafide_count}")
print(f"Spoof:    {spoof_count}")
print(f"Total:    {bonafide_count + spoof_count}")
print(f"Location: {OUTPUT_DIR}")