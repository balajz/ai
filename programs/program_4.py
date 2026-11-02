import pandas as pd

from pgmpy.models import DiscreteBayesianNetwork

# pgmpy 1.1: `MaximumLikelihoodEstimator` is deprecated, and `fit` now needs an
# initialized estimator instance. `DiscreteMLE()` is its replacement.
from pgmpy.parameter_estimator import DiscreteMLE

print("=" * 70)
print("LEARNING BAYESIAN PARAMETERS USING MLE")
print("=" * 70)

# Create training dataset
data = pd.DataFrame({
    'Disease': [
        1, 1, 1, 0, 0, 1, 0, 1, 0, 0,
        1, 0, 1, 0, 1, 0, 1, 1, 0, 0
    ],

    'Fever': [
        1, 1, 1, 0, 0, 1, 0, 1, 0, 0,
        1, 0, 1, 0, 1, 1, 0, 1, 0, 0
    ],

    'Cough': [
        1, 0, 1, 0, 1, 1, 0, 1, 0, 1,
        1, 0, 1, 0, 1, 1, 1, 0, 0, 0
    ],

    'Diagnosis': [
        1, 1, 1, 0, 0, 1, 0, 1, 0, 0,
        1, 0, 1, 0, 1, 0, 1, 0, 0, 0
    ]
})

print("\nTraining Dataset\n")
print(data)

# Define Bayesian Network structure
model = DiscreteBayesianNetwork([
    ('Disease', 'Fever'),
    ('Disease', 'Cough'),
    ('Fever', 'Diagnosis'),
    ('Cough', 'Diagnosis')
])

print("\n")
print("=" * 70)
print("Learning Parameters...")
print("=" * 70)

# Learn parameters using Maximum Likelihood Estimation
model.fit(
    data,
    estimator=DiscreteMLE()
)

print("\nModel Learning Completed Successfully\n")

print("=" * 70)
print("LEARNED CONDITIONAL PROBABILITY TABLES")
print("=" * 70)

for cpd in model.get_cpds():
    print(cpd)
    print()

print("=" * 70)
print("Model Validation")
print("=" * 70)

print(model.check_model())

print("\nExperiment Completed Successfully")
