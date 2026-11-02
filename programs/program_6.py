import numpy as np
from hmmlearn import hmm

# Show all matrices with a fixed 4-decimal format, like the book's output.
np.set_printoptions(precision=4, suppress=True, floatmode="fixed")

print("=" * 70)
print("EM ALGORITHM (BAUM-WELCH) FOR HIDDEN MARKOV MODEL")
print("=" * 70)

# -------------------------------------------------------
# Observation Sequence
# -------------------------------------------------------
# 0 = Walk, 1 = Shop, 2 = Clean
observations = np.array([
    [0], [1], [2], [0], [1],
    [2], [1], [0], [2], [1]
])

print("\nObservation Sequence\n")
print(observations.ravel())

# -------------------------------------------------------
# Create HMM Model
# -------------------------------------------------------
# init_params="" is the important part: it tells hmmlearn to KEEP the initial
# matrices set below as the starting point for EM. With the default
# init_params="ste", fit() silently re-initializes them with random values.
model = hmm.CategoricalHMM(
    n_components=2,
    n_iter=100,
    random_state=42,
    n_features=3,
    init_params=""
)

# -------------------------------------------------------
# Initial Parameters
# -------------------------------------------------------
model.startprob_ = np.array([0.6, 0.4])
model.transmat_ = np.array([
    [0.7, 0.3],
    [0.4, 0.6]
])
model.emissionprob_ = np.array([
    [0.6, 0.3, 0.1],
    [0.1, 0.4, 0.5]
])

print("\nTraining HMM using EM Algorithm...\n")

# -------------------------------------------------------
# Train Model
# -------------------------------------------------------
model.fit(observations)

# -------------------------------------------------------
# Display Learned Parameters
# -------------------------------------------------------
print("=" * 70)
print("OPTIMIZED START PROBABILITIES")
print("=" * 70)
print(np.round(model.startprob_, 4))

print("\n")
print("=" * 70)
print("OPTIMIZED TRANSITION MATRIX")
print("=" * 70)
print(np.round(model.transmat_, 4))

print("\n")
print("=" * 70)
print("OPTIMIZED EMISSION MATRIX")
print("=" * 70)
print(np.round(model.emissionprob_, 4))

print("\n")
# -------------------------------------------------------
# Log Likelihood
# -------------------------------------------------------
score = model.score(observations)
print("=" * 70)
print("LOG LIKELIHOOD")
print("=" * 70)
print(round(score, 4))

print("\nExperiment Completed Successfully")
