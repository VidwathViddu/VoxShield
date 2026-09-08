SPOOF_THRESHOLD = 0.50
SPEAKER_THRESHOLD = 0.65


def calculate_risk(spoof_probability, speaker_similarity):
    spoofed = spoof_probability >= SPOOF_THRESHOLD
    matched = speaker_similarity >= SPEAKER_THRESHOLD

    if not spoofed and matched:
        return {
            "risk_level": "LOW",
            "decision": "AUTHENTIC_VOICE"
        }

    elif spoofed and matched:
        return {
            "risk_level": "HIGH",
            "decision": "POSSIBLE_VOICE_CLONING"
        }

    elif spoofed and not matched:
        return {
            "risk_level": "HIGH",
            "decision": "SPOOFED_VOICE_DETECTED"
        }

    else:
        return {
            "risk_level": "MEDIUM",
            "decision": "POSSIBLE_SPEAKER_MISMATCH"
        }