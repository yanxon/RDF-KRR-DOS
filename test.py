import glob, os
import numpy as np
from pymatgen import MPRester
from RDF import *

m = MPRester('ZkyR13aTi9z5hLbX')
datas = m.query(criteria = {#"band_gap": 0.0,
                            'icsd_ids.0': {'$exists': True}
                            },
               properties=["pretty_formula", "material_id", "formula", "spacegroup.symbol", "volume", "icsd_ids"])

X = []         # RDF
Y = []         # DOS
sg_ICSD_PRB = []        # space group
compound_ICSD_PRB = []  # compound
mp_ids = []

# Grab all the ICSD id from the PRB data
os.chdir('prb_data')
ICSD_IDs = []
for file in glob.glob('*.dos'):
    ICSD_IDs.append(int(file[:-4]))
        
for ID in ICSD_IDs:
    # Look for each PRB ID in data screened in MP database
#    try:
    for data in datas:
        if ID in data['icsd_ids']:
            break
    
    y = []
    for line in open(str(ID)+'.dos','r'):
        y.append(float(line))
    y = np.asarray(y) / data['volume']
        
    structure = m.get_structure_by_material_id(data['material_id'])
    struc = RDF(structure)
    X.append(struc.RDF[1,:])
    Y.append(y)
#        mp_ids.append(data['material_id'])
#        sg_ICSD_PRB.append(data['spacegroup.symbol'])
#        compound_ICSD_PRB.append(data['pretty_formula'])
        
    print(data['material_id'], ' is stored')
        
#    except:
#        pass

np.savetxt('X_ICSD_PRB.txt', X)
np.savetxt('Y_ICSD_PRB.txt', Y)
np.savetxt('sg_ICSD_PRB.txt', sg_ICSD_PRB)
np.savetxt('compound_ICSD_PRB.txt', compound_ICSD_PRB)
np.savetxt('mp_id.txt', mp_ids)
