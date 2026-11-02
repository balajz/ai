import pandas as pd

print("=" * 70)
print(" DECISION NETWORK - MEDICAL DIAGNOSIS SUPPORT SYSTEM ")
print("=" * 70)

# ----------------------------------------------------------
# Disease Probabilities
# ----------------------------------------------------------
disease_probability = {
    "Flu": 0.60,
    "Common Cold": 0.25,
    "No disease": 0.15
}

# ----------------------------------------------------------
# Utility Table
# ----------------------------------------------------------
utility = {
    "Flu Diagnosis": {
        "Flu": 12000,
        "Common Cold": 7000,
        "No disease": -6000
    },
    "Healthy Diagnosis": {
        "Flu": 9000,
        "Common Cold": 6500,
        "No disease": 1000
    }
}

# ----------------------------------------------------------
# Calculate Expected Utility
# ----------------------------------------------------------
expected_utility = {}

print("\nDisease Probabilities\n")

for disease, prob in disease_probability.items():
    print(f"{disease:20s} : {prob}")

print("\n")

for decision, outcomes in utility.items():
    eu = 0
    for disease in disease_probability:
        eu += disease_probability[disease] * outcomes[disease]
    expected_utility[decision] = eu

# ----------------------------------------------------------
# Display Utility Table
# ----------------------------------------------------------
print("=" * 70)
print("EXPECTED UTILITY OF EACH DECISION")
print("=" * 70)

df = pd.DataFrame(
    expected_utility.items(),
    columns=["Decision", "Expected Utility"]
)
print(df)

print()

# ----------------------------------------------------------
# Best Decision
# ----------------------------------------------------------
best = max(expected_utility, key=expected_utility.get)

print("=" * 70)
print("OPTIMAL DECISION")
print("=" * 70)
print("Best Decision :", best)
print("Maximum Expected Utility :", round(expected_utility[best], 2))

print()
print("=" * 70)
print("Decision Analysis")
print("=" * 70)

for d, v in expected_utility.items():
    print(f"{d:20s} --> {v:.2f}")

print()
print("Experiment Completed Successfully")
