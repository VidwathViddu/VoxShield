"""Print ECAPA cosine scores for labelled audio pairs.

Example:
python evaluate_pairs.py reference.wav same_speaker.wav different_speaker.wav
"""

from __future__ import annotations

import argparse

from speaker_verification.service import verify_speaker


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare same- and different-speaker recordings")
    parser.add_argument("reference", help="Reference-speaker recording")
    parser.add_argument("same_speaker", help="Another recording of the reference speaker")
    parser.add_argument("different_speaker", help="Recording of another speaker")
    args = parser.parse_args()

    print("Same speaker:     ", verify_speaker(args.reference, args.same_speaker))
    print("Different speaker:", verify_speaker(args.reference, args.different_speaker))


if __name__ == "__main__":
    main()
