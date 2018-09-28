import glob, os
import numpy as np
from pymatgen import MPRester
from RDF import *

m = MPRester('ZkyR13aTi9z5hLbX')
datas = m.query(criteria = {#"band_gap": 0.0,
                            'icsd_ids.0': {'$exists': True}
                            },
               properties=["pretty_formula", "material_id", "formula", "spacegroup.symbol", "icsd_ids"])

X_ICSD_PRB = []         # RDF
Y_ICSD_PRB = []         # DOS
sg_ICSD_PRB = []        # space group
compound_ICSD_PRB = []  # compound
mp_id = []

# Grab all the ICSD id from the PRB data
os.chdir('prb_data')
ICSD_IDs = []
for file in glob.glob('*.dos'):
    ICSD_IDs.append(int(file[:-4]))
        
for ID in ICSD_IDs:
    # Look for each PRB ID in data screened in MP database
    try:
        for data in datas:
            if ID in data['icsd_ids']:
                mp_id.append(data['material_id'])
                break
    
        y = []
        for line in open(str(ID)+'.dos','r'):
            y.append(float(line))
        
        structure = m.get_structure_by_material_id(data['material_id'])
        struc = RDF(structure)
        X_ICSD_PRB.append(struc.RDF[1,:])
        Y_ICSD_PRB.append(y)
        sg_ICSD_PRB.append(data['spacegroup.symbol'])
        compound_ICSD_PRB.append(data['pretty_formula'])
        
        print(data['material_id'], ' is stored')
    except:
        pass

np.savetxt('X_ICSD_PRB.txt', X_ICSD_PRB)
np.savetxt('Y_ICSD_PRB.txt', Y_ICSD_PRB)
np.savetxt('sg_ICSD_PRB.txt', sg_ICSD_PRB)
np.savetxt('compound_ICSD_PRB.txt', compound_ICSD_PRB)
