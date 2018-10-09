# Gradient Boosting Regression with Random Forest algorithm
# is employed to predict Formation Energy of materials with
# RDF as the descriptor. Dataset is taken from Jarvis.

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from Descriptors.RDF import *
from pymatgen.core.structure import Structure
from monty.json import MontyEncoder, MontyDecoder
from monty.serialization import loadfn, dumpfn
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import numpy as np
import sys

# Functions
def isfloat(value):
    #Simple function to check if the data is available/float
    try:
        float(value)
        return True
    except ValueError:
        return False

def get_features(data):
    X = []  #RDF
    Y = []  #Formation energy
    for i in data:
        y = i['form_enp']
        if isfloat(y):
            crystal = i['final_str']
            X.append(RDF(crystal).RDF[1,:])
            Y.append(y)
    return X, Y

# Import data
data = loadfn('Datasets/jdft_3d-7-7-2018.json',cls=MontyDecoder)

# Split to train and test sets
X, Y = get_features(data)
X=np.array(X).astype(np.float)
Y=np.array(Y).astype(np.float)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state=0)

# Perform gradient boosting
est= GradientBoostingRegressor(loss = 'huber', learning_rate = 0.01, n_estimators = 200)
pipe=Pipeline([("fs", VarianceThreshold()),("est", est)])
pipe.fit(X_train,Y_train)

# Test set
total_mae = 0   # Total mean absolute error
n = 0           # Number of test points
Y_predicted = [] # for plotting purpose

for i in range(len(Y_test)):
    y_predicted = pipe.predict([X_test[i]])[0]
    mae = abs(y_predicted - Y_test[i])
    total_mae += mae
    n += 1

    Y_predicted.append(y_predicted)

print(total_mae/n)

# Plotting

plt.plot(Y_test, Y_predicted, 'bo')
plt.xlabel('Enthalpy_dft (eV/atom)')
plt.ylabel('Enthalpy_ML (eV/atom)')
plt.savefig('Results/enthalpy_form.png')
