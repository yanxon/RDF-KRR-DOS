# In this script, metallic materials is mined from Materials Project (MP) database.
# Structural and density of states (DOS) information is to be mined.
# From the structure, radial distribution function (RDF) can be constructed using the RDF code.
# DOS at fermi level can also be extracted.


# Importing libraries
import numpy as np
from pymatgen import MPRester
from RDF import *

# Input your MP ID
m = MPRester('ZkyR13aTi9z5hLbX')

# Data filtering: metal, i.e. band gap = 0.0.
datas = m.query(criteria = {"band_gap": 0.0,
                            "nelements": {"$lt": 4},
                            'icsd_ids.0': {'$exists': True}
                            }, 
               properties=["pretty_formula", "material_id", "formula", "spacegroup.symbol"])

# Arrays to store chemical formulas and MP's materials ID.
formula = []
ids = []

points = len(ids)

for data in datas:
    formula.append(data['pretty_formula'])
    ids.append(data['material_id'])

# Appending RDF and 
X = [] # Reserve for RDF structure; X is 1D array.
Y = [] # Reserve for DOS at fermi level; Y is 1D array.
j = 0

for i, identity in enumerate(ids):
    try:
        # Mining DOS
        dos = m.get_dos_by_material_id(identity)
        energies = dos.energies - dos.efermi
        total_dos = sum(dos.densities.values()) # Sum both spins, if present
        
        combine = np.vstack((energies, total_dos))
        
        # Extracting the DOS at fermi level.
        combine_abs = abs(combine[:,0])
        find_ele_at_fermi = np.where(combine_abs == min(combine_abs))
        ele_at_fermi = find_ele_at_fermi[0][0]

        # Constructing RDF
        structure = m.get_structure_by_material_id(identity)
        struc = RDF(structure)
        X.append(struc.RDF[:,1])
        Y.append(combine[ele_at_fermi,1])
        		
        print('Progress: ', i+1, '/', points)
        
    except:
        del ids[i] #del the materials that don't have DOS in MP database
        print('Progress: ', i+1, '/', points, ' is passed')
        
np.savetxt('C:/Resources/Install/plugins/data/X.txt', X)
np.savetxt('C:/Resources/Install/plugins/data/Y.txt', Y)
