def calculate_risk(spoof_probability, speaker_similarity):
    """
    Calculate the impersonation risk using:
    - spoof_probability: probability that the voice is AI-generated
    - speaker_similarity: similarity to the claimed/reference speaker
    """

    if spoof_probability >= 0.75 and speaker_similarity < 0.50:
        risk_level = "HIGH"
        decision = "POSSIBLE_IMPERSONATION"

    elif spoof_probability >= 0.75:
        risk_level = "HIGH"
        decision = "POSSIBLE_AI_GENERATED_VOICE"

    elif speaker_similarity < 0.50:
        risk_level = "MEDIUM"
        decision = "POSSIBLE_SPEAKER_MISMATCH"

    else:
        risk_level = "LOW"
        decision = "VOICE_APPEARS_AUTHENTIC"

    return {
        "risk_level": risk_level,
        "decision": decision
    }