import numpy as np

print("HIDDEN MARKOV MODEL - MEDICAL DIAGNOSIS")
print("=" * 70)

# Define hidden states
states = ["Mild Infection", "Severe Infection"]

# Define observations
observations = ["Fever", "Cough", "Fatigue"]

# Observation sequence
obs_sequence = [0, 1, 2]

# Initial state probabilities
start_probability = np.array([0.5, 0.5])

# Transition probability matrix
transition_probability = np.array([
    [0.7, 0.3],
    [0.4, 0.6]
])

# Emission probability matrix
emission_probability = np.array([
    [0.6, 0.3, 0.1],
    [0.1, 0.4, 0.5]
])

# Display model information
print("Hidden States:", states)
print("Observations:", observations)

print("\nInitial Probabilities")
print(start_probability)

print("\nTransition Probability Matrix")
print(transition_probability)

print("\nEmission Probability Matrix")
print(emission_probability)


# Forward Algorithm
def forward_algorithm(obs, start, trans, emit):

    n_states = len(start)

    alpha = np.zeros((len(obs), n_states))

    # Initialization
    alpha[0] = start * emit[:, obs[0]]

    # Recursion
    for t in range(1, len(obs)):
        for j in range(n_states):

            alpha[t][j] = np.sum(
                alpha[t - 1] * trans[:, j]
            ) * emit[j][obs[t]]

    # Calculate final probability
    probability = np.sum(alpha[-1])

    return alpha, probability


# Execute Forward Algorithm
alpha, probability = forward_algorithm(
    obs_sequence,
    start_probability,
    transition_probability,
    emission_probability
)

print("\n")
print("=" * 70)
print("RESULT")
print("=" * 70)

print(
    "Observation Sequence:",
    ["Fever", "Cough", "Fatigue"]
)

print(
    "Likelihood of Observation Sequence:",
    round(probability, 5)
)

print("Experiment Completed Successfully")
