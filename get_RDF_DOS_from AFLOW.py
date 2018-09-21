"""
This code mines density of states (DOS) data from AFLOW database.
AFLOW has DOS info stored in a .xz zipfile.

"""

# Import libraries
from aflow import *
import lzma
import numpy as np
import pandas as pd
from RDF import *

def save_xz(filename, URL):
    """
    1. Save .xz zipfile downloaded from an online database.
    2. Unzip the zipped files.
    
    Args:
        URL: provide a URL of the database to look for the zipfile.

        filename: provide the name of the file.
                  filename should end with .xz
    """
    URL(filename)
    zipfile = lzma.LZMAFile(filename).read()
    newfilepath = filename[:-3]
    fo = open(newfilepath+'.txt', 'wb').write(zipfile)
    
def get_DOS_fermi(filename):
    """
    This function takes DOS file and return the intensity at the fermi level.
    
    Args:
        filename: provide the DOS file
        
    Return:
        DOS at fermi level
    """
    with open(filename, 'r') as fin:
        dataf = fin.read().splitlines(True)
        fin.close()
    with open(filename, 'w') as fout:
        E_Fermi = [float(i) for i in dataf[5].split()][3]
        fout.writelines(dataf[6:5006])
        fout.close()
        
    DOS = np.genfromtxt(filename, dtype = float)
    energy = DOS[:,0] - E_Fermi
    dos = DOS[:,1]
    combine = np.vstack((energy, dos))
    combine_abs = abs(combine[0,:])
    find_ele_at_fermi = np.where(combine_abs == min(combine_abs))
    ele_at_fermi = find_ele_at_fermi[0][0]
    
    return combine[1,ele_at_fermi]

# Get materials from AFLOW database based on the given criteria: 
# metal and no more than 3 different elements.
results = search(batch_size = 100
                ).filter(K.Egap_type == 'metal'
                ).filter(K.nspecies < 4)

n = len(results) # number of avaiable data points

X = [] # RDF of materials
Y = [] # Density of states at fermi level

for i, result in enumerate(results[2:5]):
    URL = result.files['DOSCAR.static.xz']
    save_xz(result.compound+'.xz', URL)
    
    Y.append(get_DOS_fermi(result.compound+'.txt'))

    # Construct RDF from POSCAR which obtained from AFLOW
    crystal = Structure.from_str(result.files['CONTCAR.relax.vasp'](), fmt='poscar')
    X.append(RDF(crystal).RDF[1,:])
    
    print('progress: ', i, '/', n, ' materials is done')

# Save RDF and DOS at fermi for Machine Learning
np.savetxt('X_test.txt', X)
np.savetxt('Y_test.txt', Y)
