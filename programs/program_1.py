# ==========================================================
# Experiment No.1
# Bayesian Belief Network for Medical Diagnosis
# ==========================================================

import matplotlib.pyplot as plt
import networkx as nx

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# ----------------------------------------------------------
# STEP 1 : Create Bayesian Network Structure
# ----------------------------------------------------------
model = DiscreteBayesianNetwork([
    ('Disease', 'Fever'),
    ('Disease', 'Cough'),
    ('Disease', 'Fatigue'),
    ('Disease', 'Headache')
])

# ----------------------------------------------------------
# STEP 2 : Define CPDs
# ----------------------------------------------------------
cpd_disease = TabularCPD(
    variable='Disease',
    variable_card=2,
    values=[[0.60],
            [0.40]]
)

cpd_fever = TabularCPD(
    variable='Fever',
    variable_card=2,
    values=[
        [0.85, 0.30],
        [0.15, 0.70]
    ],
    evidence=['Disease'],
    evidence_card=[2]
)

cpd_cough = TabularCPD(
    variable='Cough',
    variable_card=2,
    values=[
        [0.75, 0.25],
        [0.25, 0.75]
    ],
    evidence=['Disease'],
    evidence_card=[2]
)

cpd_fatigue = TabularCPD(
    variable='Fatigue',
    variable_card=2,
    values=[
        [0.80, 0.20],
        [0.20, 0.80]
    ],
    evidence=['Disease'],
    evidence_card=[2]
)

cpd_headache = TabularCPD(
    variable='Headache',
    variable_card=2,
    values=[
        [0.80, 0.35],
        [0.20, 0.65]
    ],
    evidence=['Disease'],
    evidence_card=[2]
)

# ----------------------------------------------------------
# STEP 3 : Add CPDs
# ----------------------------------------------------------
model.add_cpds(
    cpd_disease,
    cpd_fever,
    cpd_cough,
    cpd_fatigue,
    cpd_headache
)

# ----------------------------------------------------------
# STEP 4 : Validate Model
# ----------------------------------------------------------
print("=" * 60)
print("Checking Bayesian Network")
print("=" * 60)
print(model.check_model())

# ----------------------------------------------------------
# STEP 5 : Draw Network
# ----------------------------------------------------------
# pgmpy stores weight=None on every edge, which makes networkx's
# spring_layout return NaN coordinates and crashes arrow drawing.
graph = nx.DiGraph()
graph.add_nodes_from(model.nodes())
graph.add_edges_from(model.edges())

plt.figure(figsize=(10, 7))

nx.draw_networkx(
    graph,
    with_labels=True,
    node_color="lightblue",
    node_size=2700,
    font_size=10,
    font_weight="bold",
    arrows=True
)

plt.title("Bayesian Belief Network for Medical Diagnosis")
plt.show()

# ----------------------------------------------------------
# STEP 6 : Exact Inference
# ----------------------------------------------------------
infer = VariableElimination(model)

print("\n")
print("=" * 60)
print("QUERY 1")
print("=" * 60)

result = infer.query(
    variables=['Disease'],
    evidence={
        'Fever': 1
    }
)

print(result)

print("\n")
print("=" * 60)
print("QUERY 2")
print("=" * 60)

result = infer.query(
    variables=['Disease'],
    evidence={
        'Fever': 1,
        'Cough': 1
    }
)

print(result)

print("\n")
print("=" * 60)
print("QUERY 3")
print("=" * 60)

result = infer.query(
    variables=['Disease'],
    evidence={
        'Fever': 1,
        'Cough': 1,
        'Fatigue': 1
    }
)

print(result)

print("\n")
print("=" * 60)
print("QUERY 4")
print("=" * 60)

result = infer.query(
    variables=['Disease'],
    evidence={
        'Fever': 1,
        'Cough': 1,
        'Headache': 1
    }
)

print(result)

print("\n")
print("=" * 60)
print("QUERY 5")
print("=" * 60)

result = infer.query(
    variables=['Disease'],
    evidence={
        'Fever': 1,
        'Cough': 1,
        'Fatigue': 1,
        'Headache': 1
    }
)

print(result)

print("\nExperiment Completed Successfully")
