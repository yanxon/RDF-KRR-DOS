# Gradient Boosting Regression with Random Forest algorithm
# is employed to predict Formation Energy of materials with
# RDF as the descriptor. Dataset is taken from Jarvis.

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from RDF import *
from get_desc import *
from pymatgen.core.structure import Structure
from monty.json import MontyEncoder, MontyDecoder
from monty.serialization import loadfn, dumpfn
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.feature_selection import VarianceThreshold
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

def get_training_set(data):
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
data = loadfn('jdft_3d-7-7-2018.json',cls=MontyDecoder)

X, Y = get_training_set(data[:20000])

# Perform gradient boosting
X=np.array(X).astype(np.float)
Y=np.array(Y).astype(np.float)
est= GradientBoostingRegressor()
pipe=Pipeline([ ("fs", VarianceThreshold()),("est", est)])
pipe.fit(X,Y)

# Test set

total_mae = 0   # Total mean absolute error
n = 0           # Number of test points
Y_predicted = [] # for plotting purpose
Y_dft = []       # for plotting purpose

for i in data[20000:]:
    crystal = i['final_str']
    test_x = RDF(crystal).RDF[1,:]
    y_predicted = pipe.predict([test_x])[0]
    y_dft = i['form_enp']
    mae = abs(y_predicted - y_dft)
    total_mae += mae
    n += 1

    Y_predicted.append(y_predicted)
    Y_dft.append(y_dft)

print(total_mae/n)

# Plotting

plt.plot(Y_dft, Y_predicted, 'bo')
plt.xlabel('Enthalpy_dft (eV/atom)')
plt.ylabel('Enthalpy_ML (eV/atom)')
plt.savefig('enthalpy_form.png')
