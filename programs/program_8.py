import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

print("=" * 60)
print("MEDICAL DISEASE DIAGNOSIS")
print("=" * 60)

# Training dataset
X = np.array([
    [1, 1, 1, 1],
    [1, 1, 0, 1],
    [1, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 1],
    [1, 1, 1, 0],
    [0, 1, 1, 0],
    [1, 0, 0, 1]
])

# Labels for disease classification
y = np.array([
    "Flu",
    "Flu",
    "Cold",
    "Cold",
    "Flu",
    "Flu",
    "Allergy",
    "Allergy"
])

# Initialize classifiers
dt = DecisionTreeClassifier(random_state=42)
nb = GaussianNB()

# Train models
dt.fit(X, y)
nb.fit(X, y)

# Enter patient symptoms
print("\nEnter Symptoms (1 = Yes, 0 = No)")

fever = int(input("Fever: "))
cough = int(input("Cough: "))
fatigue = int(input("Fatigue: "))
headache = int(input("Headache: "))

# Create patient data
patient = np.array([
    [fever, cough, fatigue, headache]
])

print("\n" + "=" * 60)
print("DIAGNOSIS RESULTS")
print("=" * 60)

# Decision Tree prediction
print(
    "Decision Tree Prediction:",
    dt.predict(patient)[0]
)

# Naive Bayes prediction
print(
    "Naive Bayes Prediction:",
    nb.predict(patient)[0]
)

# Display selected symptoms
print("\nSelected Symptoms")

symptoms = [
    "Fever",
    "Cough",
    "Fatigue",
    "Headache"
]

for name, value in zip(symptoms, patient[0]):
    if value == 1:
        print(" -", name)

print("\nExperiment Completed Successfully")
