import glob, os
import numpy as np
from pymatgen import MPRester
from RDF import *

def frequency(a, x):
    count = 0
     
    for i in a:
        if i == x: count += 1
    return count

m = MPRester('ZkyR13aTi9z5hLbX')
datas = m.query(criteria = {#"band_gap": 0.0,
                            'icsd_ids.0': {'$exists': True}
                            },
               properties=["pretty_formula", "material_id", "formula", "spacegroup.symbol", "icsd_ids"])

X = []
Y = []
mp_id = []
os.chdir('prb_data')

ICSD_IDs = []
for file in glob.glob('*.dos'):
    ICSD_IDs.append(int(file[:-4]))

#ICSD_IDs = np.reshape(ICSD_IDs, (3822,1))

#for data in datas:
#    for ID in data['icsd_ids']:
#        if ID in ICSD_IDs:
#            print(ID, data['icsd_ids'])
#            mp_id.append(data['material_id'])

print(len(mp_id))

#for data in datas:
#    for ID in data['icsd_ids']:
#        if frequency(data['icsd_ids'],ID) != 1:
#            print(ID,data['icsd_ids'],data['material_id'])
#            
#for ID in ICSD_IDs:
#    if frequency(ICSD_IDs,ID) != 1:
#        print('aha!')
        
for ID in ICSD_IDs:
    for data in datas:
        if ID in data['icsd_ids']:
            mp_id.append(data['material_id'])
            break
    
    y = []
    for line in open(str(ID)+'.dos','r'):
        y.append(float(line))
    Y.append(y)
    structure = m.get_structure_by_material_id(data['material_id'])
    struc = RDF(structure)
    X.append(struc.RDF[1,:])
    print(data['material_id'], ' is pass')

## more efficient than the for loop above:
#for data in datas:
#    add = False
#    for id in data['icsd_ids']:
#        if id in ICSD_IDs:
#            add = True
#            break
#    if add:
#        mp_id.append(data['material_id'])
#        y = []
#        for line in open(str(id)+'.dos','r'):
#            y.append(float(line))
#        Y.append(y)
#        structure = m.get_structure_by_material_id(data['material_id'])
#        struc = RDF(structure)
#        X.append(struc.RDF[1,:])
#        print(data['material_id'], ' is pass')

np.savetxt('X_.txt',X)
np.savetxt('Y_.txt',Y)
