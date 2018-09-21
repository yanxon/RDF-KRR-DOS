# Explain code


# Import libraries
from aflow import *
import lzma
import numpy as np
import pandas as pd
from RDF import *


# Get materials from AFLOW database based on the given criteria: 
# metal and no more than 3 different elements.
results = search(batch_size = 100
                ).filter(K.Egap_type == 'metal'
                ).filter(K.nspecies < 4)

n = len(results) # number of avaiable data points


X = [] # RDF of materials
Y = [] # Density of states at fermi level

for i, result in enumerate(results[2:5]):
    # AFLOW has DOS info in a zip file (.xz).
    # The following lines are to extract DOS from the zip file and
    # to store the DOS info in your computer.
    path = 'C:/Users/Carbon/Desktop/Aflow_DOS/' + result.compound + '.xz'
    result.files['DOSCAR.static.xz'](path)
    zipfile = lzma.LZMAFile(path).read()
    newfilepath = path[:-4]
    fo = open(newfilepath+'.txt', 'wb').write(zipfile)

    with open(newfilepath+'.txt', 'r') as fin:
        dataf = fin.read().splitlines(True)
        fin.close()

    with open(newfilepath+'.txt', 'w') as fout:
        E_Fermi = [float(i) for i in dataf[5].split()][2]   #Extract Fermi energy
        fout.writelines(dataf[6:5006])
        fout.close()
    
    # Extract DOS at fermi level from the saved file.
    DOS = np.genfromtxt(newfilepath+'.txt', dtype = float)
    energy = DOS[:,0] - E_Fermi                             # E - Efermi
    dos = DOS[:,1]
    combine = np.vstack((energy, dos))
    combine_abs = abs(combine[0,:])
    find_ele_at_fermi = np.where(combine_abs == min(combine_abs))
    ele_at_fermi = find_ele_at_fermi[0][0]
    Y.append(combine[1,ele_at_fermi])

    # Construct RDF from POSCAR which obtained from AFLOW
    crystal = Structure.from_str(result.files['CONTCAR.relax.vasp'](), fmt='poscar')
    X.append(RDF(crystal).RDF[1,:])

# Save RDF and DOS at fermi for Machine Learning
np.savetxt('X_test.txt', X)
np.savetxt('Y_test.txt', Y)
