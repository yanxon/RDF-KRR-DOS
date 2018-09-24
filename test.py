import glob, os
import numpy as np
from pymatgen import MPRester


os.chdir('prb_data')

ICSD_IDs = []

for file in glob.glob('*.dos'):
    ICSD_IDs.append(int(file[:-4])
    
ICSD_IDs = np.reshape(ICSD_IDs, (3823,1))

print(ICSD_IDs[:10])


m = MPRester('ZkyR13aTi9z5hLbX')
datas = m.query(criteria = {"band_gap": 0.0,
                            'icsd_ids.0': {'$exists': True}
                            }, 
               properties=["pretty_formula", "material_id", "formula", "spacegroup.symbol", "icsd_ids"])

X = []
Y = []

for ids in ICSD_IDs[0:1000]:
    for data in datas:
        if ids in data["icsd_ids"]:
            print('yes')

for data in datas[:10]:
    print(data['icsd_ids'])

datas[0]['icsd_ids'] in ICSD_IDs

ICSD_IDs.append('25316')