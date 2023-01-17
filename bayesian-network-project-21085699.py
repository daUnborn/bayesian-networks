#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 12:31:18 2022

@author: camaike
"""

from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination
import pandas as pd
import bnlearn as bn
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz
import graphviz


# Load the data
columns = ['Clump Thickness',
'Uniformity of Cell Size',
'Uniformity of Cell Shape',
'Marginal Adhesion',
'Single Epithelial Cell Size',
'Bare Nuclei',
'Bland Chromatin',
'Normal Nucleoli',
'Mitoses',
'Class']

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',
names=columns)

print(df)

#Data evaluation
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum()) # there are no null values in the data

#define the edges
edges = [('Class', 'Clump Thickness'),
                      ('Class', 'Uniformity of Cell Size'),
                      ('Class', 'Uniformity of Cell Shape'),
                      ('Class', 'Marginal Adhesion'),
                      ('Class', 'Single Epithelial Cell Size'),
                      ('Class', 'Bare Nuclei'),
                      ('Class', 'Bland Chromatin'),
                      ('Class', 'Normal Nucleoli'),
                      ('Class', 'Mitoses')]

## Specify the network structure by creating edges between the nodes
model = BayesianModel()

#adding data columns wise to model
for col in df.columns:
    model.add_node(col)
    if col != "Class":
        model.add_edge("Class", col)


# Estimate the parameters of the network using the dataset and fitting model with data
model.fit(df, estimator=MaximumLikelihoodEstimator)


# Use the network for classification tasks by using variable elimination imported in staring
# Then, finding the probability of each class by using query function
infer = VariableElimination(model)


Clump_Thickness = 5
Uniformity_of_Cell_Size = 1
Uniformity_of_Cell_Shape = 1
Marginal_Adhesion = 1
Bare_Nuclei = 2
Bland_Chromatin = 1
Normal_Nucleoli = 3
Mitoses = 2
Single_Epithelial_Cell_Size = 1


probability = infer.query(['Class'], evidence={'Clump Thickness':Clump_Thickness})
print(f"Probability when Clump Thickness is {Clump_Thickness} is shown below.")
print(probability)

probability = infer.query(['Class'], evidence={'Uniformity of Cell Size':Uniformity_of_Cell_Size})
print(f"Probability when Uniformity of Cell Size is {Uniformity_of_Cell_Size} is shown below.")
print(probability)

probability = infer.query(['Class'], evidence={'Uniformity of Cell Shape':Uniformity_of_Cell_Shape})
print(f"Probability when Uniformity of Cell Shape is {Uniformity_of_Cell_Shape} is shown below.")
print(probability)


probability = infer.query(['Class'], evidence={'Marginal Adhesion':Marginal_Adhesion})
print(f"Probability when Marginal Adhesion is {Marginal_Adhesion} is shown below.")
print(probability)

'''

probability = infer.query(['Class'], evidence={'Bare Nuclei':Bare_Nuclei})
print(f"Probability when Bare Nuclei is {Bare_Nuclei} is shown below.")
print(probability)

probability = infer.query(['Class'], evidence={'Bland Chromatin':Bland_Chromatin})
print(f"Probability when Bland Chromatin is {Bland_Chromatin} is shown below.")
print(probability)

probability = infer.query(['Class'], evidence={'Normal Nucleoli':Normal_Nucleoli})
print(f"Probability when Normal Nucleoli is {Normal_Nucleoli} is shown below.")
print(probability)

probability = infer.query(['Class'], evidence={'Mitoses':Mitoses})
print(f"Probability when Mitoses is {Mitoses} is shown below.")
print(probability)

probability = infer.query(['Class'], evidence={'Single Epithelial Cell Size':Single_Epithelial_Cell_Size})
print(f"Probability when Single Epithelial Cell Size is {Single_Epithelial_Cell_Size} is shown below.")
print(probability)
'''


DAG = bn.make_DAG(edges)
# [BNLEARN] Bayesian DAG created.

# Print the CPDs
CPDs = bn.print_CPD(DAG)
# [BNLEARN.print_CPD] No CPDs to print. Use bnlearn.plot(DAG) to make a plot.

bn.plot(DAG)


#########################################################################################
# While this might be a bit abstrat
#An attempt to use DecisionTreeClassifer to classify cancer based on some attributes
# Also, a decision tree is generated showing the gini index of each attributes on the tree
######################################################################################



# Load the data
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',
                 names=columns)

# Replace missing values with -99999
df.replace('?', -99999, inplace=True)

# Get the data and the labels
X = df.drop(['Class'], axis=1).values
y = df['Class'].values

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create the decision tree bayesian model
clf = DecisionTreeClassifier()

# Train the model
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Print the accuracy score
print("Accuracy: ", accuracy_score(y_test, y_pred))

# Define the new instance
new_instance = np.array([[5, 1, 1, 1, 2, 1, 3, 1, 1]])

# Predict the class of the new instance
predicted_class = clf.predict(new_instance)

# Print the predicted class
print("Predicted class: ", predicted_class)

# Create the decision tree bayesian model
clf = DecisionTreeClassifier()

# Train the model
clf.fit(X_train, y_train)

# Export the tree to a dot file
dot_data = export_graphviz(clf, out_file=None, feature_names=columns[:-1], class_names=["Benign", "Malignant"])

# Create a graphviz object
graph = graphviz.Source(dot_data)

# Display the graph
graph

