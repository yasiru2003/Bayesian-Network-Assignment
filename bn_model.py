import pandas as pd
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# loding the data from local csv file
data = pd.read_csv("lung_cancer.csv")

# changin yes/no to 1 and 0
data['LUNG_CANCER'] = data['LUNG_CANCER'].map({'YES': 1, 'NO': 0})

# fix other colums too
cols = ['SMOKING', 'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH', 'CHEST PAIN', 'ANXIETY']
for c in cols:
    data[c] = data[c].map({2: 1, 1: 0})

# selecting only 6 nodes for the assignmnt
data = data[['SMOKING', 'ANXIETY', 'LUNG_CANCER', 'COUGHING', 'SHORTNESS OF BREATH', 'CHEST PAIN']]
data.columns = ['Smoking', 'Pollution', 'Lung_Disease', 'Cough', 'Breathing_Problem', 'Chest_Pain']

data = data.dropna().astype(int)

# defining the network structure
model = DiscreteBayesianNetwork([
    ('Smoking', 'Lung_Disease'),
    ('Pollution', 'Lung_Disease'),
    ('Lung_Disease', 'Cough'),
    ('Lung_Disease', 'Breathing_Problem'),
    ('Lung_Disease', 'Chest_Pain'),
    ('Smoking', 'Cough')
])

# lerning parameters from the data
model.fit(data, estimator=MaximumLikelihoodEstimator)

# doing inferense now
infer = VariableElimination(model)

# high risk case
res1 = infer.query(variables=['Lung_Disease'], evidence={'Smoking': 1, 'Pollution': 1})
print("\nHigh Risk (Smoker=Yes, Pollution=Yes):")
print(res1)

# low risk case
res2 = infer.query(variables=['Lung_Disease'], evidence={'Smoking': 0, 'Pollution': 0})
print("\nLow Risk (Smoker=No, Pollution=No):")
print(res2)

# saving cpts for report
with open("cpts.txt", "w") as f:
    for cpd in model.get_cpds():
        f.write(f"CPD for {cpd.variable}:\n")
        f.write(str(cpd) + "\n\n")

print("Process completed")
