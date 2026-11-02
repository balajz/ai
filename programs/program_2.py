from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import ApproxInference

# ----------------------------------------------------------
# Create Bayesian Network
# ----------------------------------------------------------
# Textbook structure (Cloudy -> Sprinkler/Rain -> WetGrass) mapped to
# medical diagnosis:
#   Cloudy     -> Disease
#   Sprinkler  -> fever
#   Rain       -> cough
#   WetGrass   -> Diagnosis
model = DiscreteBayesianNetwork([
    ('Disease', 'fever'),
    ('Disease', 'cough'),
    ('fever', 'Diagnosis'),
    ('cough', 'Diagnosis')
])

# ----------------------------------------------------------
# Define CPDs
# ----------------------------------------------------------
cpd_disease = TabularCPD(
    variable='Disease',
    variable_card=2,
    values=[[0.6], [0.4]]
)

cpd_fever = TabularCPD(
    variable='fever',
    variable_card=2,
    values=[
        [0.85, 0.30],
        [0.15, 0.70]
    ],
    evidence=['Disease'],
    evidence_card=[2]
)

cpd_cough = TabularCPD(
    variable='cough',
    variable_card=2,
    values=[
        [0.75, 0.25],
        [0.25, 0.75]
    ],
    evidence=['Disease'],
    evidence_card=[2]
)

cpd_diagnosis = TabularCPD(
    variable='Diagnosis',
    variable_card=2,
    values=[
        [0.98, 0.65, 0.70, 0.10],
        [0.02, 0.35, 0.30, 0.90]
    ],
    evidence=['fever', 'cough'],
    evidence_card=[2, 2]
)

model.add_cpds(
    cpd_disease,
    cpd_fever,
    cpd_cough,
    cpd_diagnosis
)

print("=" * 60)
print("Checking Bayesian Network")
print("=" * 60)
print(model.check_model())

# ----------------------------------------------------------
# Approximate Inference
# ----------------------------------------------------------
infer = ApproxInference(model)

# pgmpy 1.1 needs the state names passed explicitly to get_distribution().
state_names = {
    cpd.variable: list(cpd.state_names[cpd.variable])
    for cpd in model.get_cpds()
}

print("\n")
print("=" * 60)
print("QUERY 1")
print("=" * 60)

samples = model.simulate(n_samples=5000, seed=42, show_progress=False)

result = infer.get_distribution(
    samples,
    variables=['cough'],
    state_names=state_names
)

print("Approximate Distribution for cough")
print(result)

print("\n")
print("=" * 60)
print("QUERY 2")
print("=" * 60)

samples = model.simulate(
    n_samples=5000,
    evidence={'Diagnosis': 1},
    seed=42,
    show_progress=False
)

result = infer.get_distribution(
    samples,
    variables=['cough'],
    state_names=state_names
)

print("Approximate Distribution for cough given Diagnosis = 1")
print(result)

print("\n")
print("=" * 60)
print("QUERY 3")
print("=" * 60)

samples = model.simulate(
    n_samples=5000,
    evidence={
        'Disease': 1,
        'Diagnosis': 1
    },
    seed=42,
    show_progress=False
)

result = infer.get_distribution(
    samples,
    variables=['fever'],
    state_names=state_names
)

print("Approximate Distribution for fever given Disease = 1 and Diagnosis = 1")
print(result)

print("\n")
print("=" * 60)
print("QUERY 4")
print("=" * 60)

samples = model.simulate(
    n_samples=5000,
    evidence={
        'Disease': 0
    },
    seed=42,
    show_progress=False
)

result = infer.get_distribution(
    samples,
    variables=['Diagnosis'],
    state_names=state_names
)

print("Approximate Distribution for Diagnosis given Disease = 0")
print(result)

print("\n")
print("=" * 60)
print("QUERY 5")
print("=" * 60)

samples = model.simulate(
    n_samples=5000,
    evidence={
        'cough': 1
    },
    seed=42,
    show_progress=False
)

result = infer.get_distribution(
    samples,
    variables=['Diagnosis'],
    state_names=state_names
)

print("Approximate Distribution for Diagnosis given cough = 1")
print(result)

print("\nExperiment Completed Successfully")
