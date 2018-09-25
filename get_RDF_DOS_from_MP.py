# Mining metal from Materials Project (MP) database (db).
# There are two information which are taken from MP db: DOS at Fermi level and RDF.
# Input: MP id.
# Output: 2 text files; DOS at Fermi level and RDF

# Importing libraries
import numpy as np
from pymatgen import MPRester
from RDF import *
from pymatgen.core.composition import Composition
from pymatgen.core.periodic_table import Element
import sys

def get_s_metal():
    """

    """
    metals = []
    for m in dir(Element)[:102]:
        ele = Element[m]
        if ele.is_alkali or ele.is_alkaline:
            metals.append(m)
    return metals

def get_p_metal():
    """

    """
    metals = []
    for m in dir(Element)[:102]:
        ele = Element[m]
        if ele.is_post_transition_metal:
            metals.append(m)
    return metals

def get_d_metal():
    """

    """
    metals = []
    for m in dir(Element)[:102]:
        ele = Element[m]
        if ele.is_transition_metal:
            metals.append(m)
    metals.append('Zr')
    return metals

sp_system = get_s_metal() + get_p_metal()
spd_system = get_s_metal() + get_p_metal() + get_d_metal()

# Input your MP ID
m = MPRester('ZkyR13aTi9z5hLbX')

# Data filtering: metal, i.e. band gap = 0.0.
datas = m.query(criteria = {"band_gap": 0.0,
                            "nelements": {"$lt": 6},
                            'icsd_ids.0': {'$exists': True}
                            },
               properties=["pretty_formula", "material_id", "elements", "spacegroup.symbol"])

# Arrays to store chemical formulas and MP's materials ID.
formula = []
ids = []

# Out of all possible metal, pick only spd system
for data in datas:
    j = 0
    for element in data['elements']:
        n_element = len(data['elements'])
        if element in spd_system:
            j += 1
    if j == len(data['elements']):
        ids.append(data['material_id'])
        formula.append(data['pretty_formula'])
        
points = len(ids)

X = [] # RDF structure; X is 1D array.
Y = [] # DOS at fermi level; Y is 1D array.

for i, identity in enumerate(ids[:100]):
    try:
        # Mining DOS
        dos = m.get_dos_by_material_id(identity)
        energies = dos.energies - dos.efermi
        total_dos = sum(dos.densities.values()) # Sum both spins, if present

        combine = np.vstack((energies, total_dos))
        
        # Extracting the DOS at fermi level.
        combine_abs = abs(combine[0,:])
        find_ele_at_fermi = np.where(combine_abs == min(combine_abs))
        ele_at_fermi = find_ele_at_fermi[0][0]

        # Constructing RDF
        structure = m.get_structure_by_material_id(identity)
        struc = RDF(structure)
        X.append(struc.RDF[1,:])
        Y.append(combine[1,ele_at_fermi-3:ele_at_fermi+4])

        print('Progress: ', i+1, '/', points, ' ---> ', formula[i])

    except:
        print('Progress: ', i+1, '/', points, ' is passed',  ' ---> ', formula[i])

np.savetxt('X_test.txt', X)
np.savetxt('Y_test.txt', Y)
