"""
This code mines density of states (DOS) data from AFLOW database.
AFLOW has DOS info stored in a .xz zipfile.
Here is the overall idea:
    1. Mine data from AFLOW in .xz zipfile
    2. Convert the .xz zipfile into .txt
    3. read DOS information from .txt
    4. append as Y (array) and save it as a textfile.
"""

# Import libraries
from aflow import *
import lzma
import numpy as np
import pandas as pd
from RDF import *
from pymatgen.core.composition import Composition
from pymatgen.core.periodic_table import Element

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
    dos = DOS[:,1]/Volume
    combine = np.vstack((energy, dos))
    combine_abs = abs(combine[0,:])
    find_ele_at_fermi = np.where(combine_abs == min(combine_abs))
    ele_at_fermi = find_ele_at_fermi[0][0]
    
    return combine[1,ele_at_fermi-3:ele_at_fermi+4]

#Y.append(combine[1,ele_at_fermi-3:ele_at_fermi+4])

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


sp_system = get_s_metal() + get_p_metal()
spd_system = get_d_metal()

# Get materials from AFLOW database based on the given criteria: 
# metal and no more than 3 different elements.
results = search(batch_size = 100
                ).filter(K.Egap_type == 'metal'
                ).filter(K.nspecies < 7)

n = len(results) # number of avaiable data points

X_all_metals = [] # RDF of materials
Y_all_metals = [] # Density of states at fermi level
sg_all_metals = [] # Space group of all metal
compounds_all_metals = [] #compounds of all metal

X_sp_metals = []
Y_sp_metals = []
sg_sp_metals = []
compounds_sp_metals = []


X_spd_metals = []
Y_spd_metals = []
sg_spd_metals = []
compounds_spd_metals = []

for i, result in enumerate(results[:7]):
    try:
        if result.catalog == 'ICSD\n':
            URL = result.files['DOSCAR.static.xz']
            save_xz(result.compound+'.xz', URL)
    
            # Construct RDF with POSCAR
            crystal = Structure.from_str(result.files['CONTCAR.relax.vasp'](), fmt='poscar')
            
#            # Appending for all metals
#            X_all_metals.append(RDF(crystal).RDF[1,:])
#            Y_all_metals.append(get_DOS_fermi(result.compound+'.txt', result))
#            sg_all_metals.append(result.spacegroup_relax)
#            compounds_all_metals.append(result.compound)
            
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
#                
#            # Appending for spd_metals
#            j = 0
#            for element in elements:
#                if element in spd_system:
#                    j += 1
#            if j == len(elements):
#                X_spd_metals.append(RDF(crystal).RDF[1,:])
#                Y_spd_metals.append(get_DOS_fermi(result.compound+'.txt', result))
#                sg_spd_metals.append(result.spacegroup_relax)
#                compounds_spd_metals.append(result.compound)
            
            print('progress: ', i+1, '/', n, ' materials is done')
            
    except:
        pass

# Save as a text file for all metals
#np.savetxt('X_all_metals.txt', X_all_metals)
#np.savetxt('Y_all_metals.txt', Y_all_metals)
#np.savetxt('sg_all_metals.txt', sg_all_metals)
#np.savetxt('compounds_all_metals.txt', compounds_all_metals, delimiter=" ", fmt="%s")

# Save as a text file for sp metals
np.savetxt('X_sp_metals.txt', X_sp_metals)
np.savetxt('Y_sp_metals.txt', Y_sp_metals)
np.savetxt('sg_sp_metals.txt', sg_sp_metals)
np.savetxt('compounds_sp_metals.txt', compounds_sp_metals, delimiter=" ", fmt="%s")
#
## Save as a text file for spd metals
#np.savetxt('X_spd_metals.txt', X_spd_metals)
#np.savetxt('Y_spd_metals.txt', Y_spd_metals)
#np.savetxt('sg_spd_metals.txt', sg_spd_metals)
#np.savetxt('compounds_spd_metals.txt', compounds_spd_metals, delimiter=" ", fmt="%s")
