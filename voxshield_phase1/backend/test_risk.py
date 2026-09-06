from risk_engine import calculate_risk


test_cases = [
    ("AI + Wrong Speaker", 0.90, 0.25),
    ("AI + Matching Speaker", 0.90, 0.85),
    ("Real + Wrong Speaker", 0.20, 0.30),
    ("Real + Matching Speaker", 0.10, 0.90),
]


for name, spoof, similarity in test_cases:
    result = calculate_risk(spoof, similarity)

    print(f"\n{name}")
    print(f"Spoof Probability: {spoof}")
    print(f"Speaker Similarity: {similarity}")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Decision: {result['decision']}")