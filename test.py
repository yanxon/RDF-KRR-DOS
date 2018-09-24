import glob, os
import numpy as np
from pymatgen import MPRester

m = MPRester('ZkyR13aTi9z5hLbX')
datas = m.query(criteria = {#"band_gap": 0.0,
                            'icsd_ids.0': {'$exists': True}
                            },
               properties=["pretty_formula", "material_id", "formula", "spacegroup.symbol", "icsd_ids"])

os.chdir('prb_data')

ICSD_IDs = []
for file in glob.glob('*.dos'):
    ICSD_IDs.append(int(file[:-4]))

ICSD_IDs = np.reshape(ICSD_IDs, (3822,1))

X = []
Y = []
mp_id = []

#for data in datas:
#    for id in ICSD_IDs:
#        if id in data["icsd_ids"]:
#            print(id, data["icsd_ids"])
#            mp_id.append(data['material_id'])

# more efficient than the for loop above:
for data in datas[0:10]:
    for id in data['icsd_ids']:
        if id in ICSD_IDs:
            print(id, data['icsd_ids']
            mp_id.append(data['material_id']

print(len(mp_id))
