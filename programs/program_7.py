import numpy as np
import random

print("=" * 70)
print("REINFORCEMENT LEARNING USING Q-LEARNING")
print("Disease Diagnosis")
print("=" * 70)

# Reward matrix
# The program uses 6 states (0..5), but the original matrix only had 5 rows,
# so R[5] / Q[5] raised IndexError as soon as the agent reached state 5.
# Row 5 is the terminal state ("Diagnosis Complete"): no outgoing actions.
R = np.array([
    [-1, 0, 0, -1, -1, -1],
    [-1, -1, -1, 0, 0, -1],
    [-1, -1, -1, -1, 0, -1],
    [-1, -1, -1, 0, -1, 100],
    [-1, -1, -1, -1, 0, 100],
    [-1, -1, -1, -1, -1, -1]
])

# Initialize Q-table
Q = np.zeros_like(R, dtype=float)

gamma = 0.8
episodes = 1000

# Fixed seed so the sampled actions (and therefore the reported path)
# are reproducible between runs.
random.seed(42)

# Q-Learning algorithm
for episode in range(episodes):

    # Start from a non-terminal state; state 5 has no valid actions,
    # so random.choice() on an empty list would fail.
    state = random.randint(0, 4)

    while True:

        # Find possible actions
        possible_actions = np.where(R[state] >= 0)[0]

        # Select a random action
        action = random.choice(possible_actions)

        # Determine next state
        next_state = action

        # Calculate future reward
        future_reward = np.max(Q[next_state])

        # Update Q-table
        Q[state, action] = (
            R[state, action] +
            gamma * future_reward
        )

        state = next_state

        if state == 5:
            break

# Display learned Q-table
print("\n")
print("=" * 70)
print("LEARNED Q-TABLE")
print("=" * 70)

print(np.round(Q, 2))

# Find optimal path
print("\n")

state = 0
path = [state]

# The step limit guards against an endless loop if a state was never visited
# (an all-zero Q row makes argmax() point at state 0).
while state != 5 and len(path) <= 20:

    next_state = int(np.argmax(Q[state]))

    path.append(next_state)

    state = next_state

# State descriptions
states = {
    0: "Patient",
    1: "Fever",
    2: "Cough",
    3: "Cold",
    4: "Flu",
    5: "Diagnosis Complete"
}

# Display optimal path
print("Optimal Diagnosis Path:")

print(" -> ".join(states[s] for s in path))

print("\nExperiment Completed Successfully")
