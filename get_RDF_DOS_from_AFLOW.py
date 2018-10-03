"""
This code mines density of states (DOS) data from AFLOW database.
AFLOW has DOS info stored in a .xz zipfile.
The overall idea:
a. Mining:
    1. Mine data from AFLOW in .xz zipfile
    2. Convert the .xz zipfile into .txt
    3. read DOS information from .txt
    4. append as Y (array) and save it as a textfile.
b. Machine Learning:
    Training with KRR
"""

# Import libraries
from aflow import *
import lzma
import numpy as np
import pandas as pd
from RDF import *
import os
from pymatgen.core.composition import Composition
from pymatgen.core.periodic_table import Element
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error

# Functions
def save_xz(filename, URL):
    """
    1. Save .xz zipfile downloaded from an online database.
    2. Unzip the zipped files.
    
    Args:
        URL: provide a URL of the database to look for the zipfile.

        filename: provide the name of the file; filename should end with '.xz'.
    """
    URL(filename)
    zipfile = lzma.LZMAFile(filename).read()
    newfilepath = filename[:-3]
    fo = open(newfilepath+'.txt', 'wb').write(zipfile)
    os.remove(filename)
    
def get_DOS_fermi(filename, volume):
    """
    This function takes DOS file and return intensities near the fermi level.
    
    Args:
        filename: provide the DOS file; filename should end with '.txt'.
        
        volume: input the material entry to include volume in the DOS.
        
    Returns:
        DOS at fermi level
    """
    with open(filename, 'r') as fin:
        dataf = fin.read().splitlines(True)
        fin.close()
    with open(filename, 'w') as fout:
        E_Fermi = [float(i) for i in dataf[5].split()][3]
        fout.writelines(dataf[6:5006])
        fout.close()
    
    Volume = volume.volume_cell
    DOS = np.genfromtxt(filename, dtype = float)
    energy = DOS[:,0] - E_Fermi
    dos = DOS[:,1]/Volume                           # 1/(eV*A^3)
    combine = np.vstack((energy, dos))
    combine_abs = abs(combine[0,:])
    find_ele_at_fermi = np.where(combine_abs == min(combine_abs))
    ele_at_fermi = find_ele_at_fermi[0][0]
    
    return combine[1,ele_at_fermi-3:ele_at_fermi+4]

def get_s_metal():
    """
    obtain all metallic elements in group 1 & 2.
    
    Returns:
        an array of metallic elements in group 1 & 2.
    """
    metals = []
    for m in dir(Element)[:102]:
        ele = Element[m]
        if ele.is_alkali or ele.is_alkaline:
            metals.append(m)
    return metals

def get_p_metal():
    """
    obtain all metallic elements in group 13 to 17.
    
    Returns:
        an array of metallic elements in group 13 to 17.
    """
    metals = []
    for m in dir(Element)[:102]:
        ele = Element[m]
        if ele.is_post_transition_metal:
            metals.append(m)
    return metals

def get_d_metal():
    """
    obtain all transition-metal elements.
    
    Returns:
        an array of transition-metal elements.
    """
    metals = []
    for m in dir(Element)[:102]:
        ele = Element[m]
        if ele.is_transition_metal:
            metals.append(m)
    metals.append('Zr')
    return metals

####################################### Part a: Mining ###########################################
# Get materials from AFLOW database based on the given criteria: 
# metal and no more than 3 different elements.

sp_system = get_s_metal() + get_p_metal()

results = search(batch_size = 100
                ).filter(K.Egap_type == 'metal'
                ).filter(K.nspecies < 7)

n = len(results) # number of avaiable data points

X_sp_metals = []
Y_sp_metals = []
sg_sp_metals = []
compounds_sp_metals = []

for i, result in enumerate(results):
    try:
        if result.catalog == 'ICSD\n':
            URL = result.files['DOSCAR.static.xz']
            save_xz(result.compound+'.xz', URL)
    
            # Construct RDF with POSCAR
            crystal = Structure.from_str(result.files['CONTCAR.relax.vasp'](), fmt='poscar')
            
            # Get elements in the compound
            elements = result.species
            last_element = elements[-1]
            last_element = last_element[:-1]
            elements[-1] = last_element
            
            # Appending for sp_metals
            j = 0
            for element in elements:
                if element in sp_system:
                    j += 1
            if j == len(elements):
                X_sp_metals.append(RDF(crystal).RDF[1,:])
                Y_sp_metals.append(get_DOS_fermi(result.compound+'.txt', result))
                sg_sp_metals.append(result.spacegroup_relax)
                compounds_sp_metals.append(result.compound)
                
            print('progress: ', i+1, '/', n, '-------- material is stored')
            
    except:
        print('progress: ', i+1, '/', n, '-------- material does not fit the criteria')
        pass

################################ Part b: Machine Learning ###################################

N_data = len(sg_sp_metals)

# Shorter RDF
X_sp_metals = X_sp_metals[:,10:30]

# Running a GridSearch to determine the best parameters
X_train, X_test, Y_train, Y_test = train_test_split(X_sp_metals, Y_sp_metals, test_size = 0.1, random_state=0)
estimator = GridSearchCV(KernelRidge(kernel='laplacian', gamma=0.1), cv=10, 
                         param_grid={"alpha": [1e6, 1e5, 1e4, 1e3, 100, 10, 1e0, 0.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6], 
                         "gamma": np.logspace(-5, 5)})
estimator.fit(X_train, Y_train)

best_alpha = estimator.best_params_['alpha']
best_gamma = estimator.best_params_['gamma']

# Train with the best parameters
estimator2 = KernelRidge(alpha = best_alpha, coef0 = 1, gamma = best_gamma, kernel='laplacian', kernel_params=None)
estimator2.fit(X_train, Y_train)
y_predicted = estimator2.predict(X_test)
r2= estimator2.score(X_test, Y_test, sample_weight=None)
print('r^2 = ', r2)

mae = mean_absolute_error(y_predicted, Y_test)
print(mae)

# Plotting
n_test = len(Y_test)

Y_testing=[]
sg_testing=[]
for i in range(n_test):
    for j in range(N_data):
        if Y_test[i,0] == Y_sp[j,0]:
            sg_testing.append(sg_sp[j])
            break

Y_test_triclinic = []
y_pred_triclinic = []
Y_test_monoclinic = []
y_pred_monoclinic = []
Y_test_orthorhombic = []
y_pred_orthorhombic = []
Y_test_tetragonal = []
y_pred_tetragonal = []
Y_test_trigonal = []
y_pred_trigonal = []
Y_test_hexagonal = []
y_pred_hexagonal = []
Y_test_cubic = []
y_pred_cubic = []

for i, sg in enumerate(sg_testing):
    if sg in [1,2]:
        Y_test_triclinic.append(Y_test[i,:])
        y_pred_triclinic.append(y_predicted[i,:])
    elif sg in np.arange(3,16):
        Y_test_monoclinic.append(Y_test[i,:])
        y_pred_monoclinic.append(y_predicted[i,:])
    elif sg in np.arange(16,75):
        Y_test_orthorhombic.append(Y_test[i,:])
        y_pred_orthorhombic.append(y_predicted[i,:])
    elif sg in np.arange(75,143):
        Y_test_tetragonal.append(Y_test[i,:])
        y_pred_tetragonal.append(y_predicted[i,:])
    elif sg in np.arange(143,168):
        Y_test_trigonal.append(Y_test[i,:])
        y_pred_trigonal.append(y_predicted[i,:])
    elif sg in np.arange(168,195):
        Y_test_hexagonal.append(Y_test[i,:])
        y_pred_hexagonal.append(y_predicted[i,:])
    elif sg in np.arange(195,231):
        Y_test_cubic.append(Y_test[i,:])
        y_pred_cubic.append(y_predicted[i,:])
        
tri = plt.scatter(y_pred_triclinic, Y_test_triclinic, c='green')
mono = plt.scatter(y_pred_monoclinic, Y_test_monoclinic, c='red')
ortho = plt.scatter(y_pred_orthorhombic, Y_test_orthorhombic, c='black')
tetra = plt.scatter(y_pred_tetragonal, Y_test_tetragonal, c='gold')
trig = plt.scatter(y_pred_trigonal, Y_test_trigonal, c='purple')
hexa = plt.scatter(y_pred_hexagonal, Y_test_hexagonal, c='darkcyan')
cub = plt.scatter(y_pred_cubic, Y_test_cubic, c='gray')

plt.title('DOS: Actual vs Predicted -- 844 crystals')
#plt.text(1.6,.3, '$r^2$: 0.6540')
plt.legend((tri, mono, ortho, tetra, trig, hexa, cub),
           ('triclinic', 'monoclinic', 'orthorhombic', 'tetragonal', 'trigonal', 'hexagonal', 'cubic'),
           loc='lower right',
           fontsize = 7)
plt.xlabel('y_predicted')
plt.ylabel('Y_test')
plt.savefig('KRR_sp_metals.png')
plt.show()
