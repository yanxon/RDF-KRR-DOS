# construct json

import os
import sys
import lzma
import numpy as np
import json
from aflow import *
from Descriptors.RDF import *
from pymatgen.core.composition import Composition
from pymatgen.core.periodic_table import Element
from pymatgen.core import Structure
from pymatgen.core import Lattice

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# Functions
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
    os.remove(filename)

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
    dos = DOS[:,1]/Volume                           # 1/(eV*A^3)
    combine = np.vstack((energy, dos))
    combine_abs = abs(combine[0,:])
                                                                                                                                                                                                                                                            1,1           Top
                    'dos_fermi': dos}
    return mat_property

####################################### Part a: Mining ###########################################
# Get materials from AFLOW database based on the given criteria:
# sp metals with less than 7 different elements.

sp_system = get_s_metal() + get_p_metal()

results = search(batch_size = 100
                ).filter(K.Egap_type == 'metal'
                ).filter(K.nspecies < 7)

n = len(results) # number of avaiable data points

X_sp_metals = []
Y_sp_metals = []
materials_info = []

for i, result in enumerate(results[51:55]):
    try:
        if result.catalog == 'ICSD\n':
            URL = result.files['DOSCAR.static.xz']
            save_xz(result.compound+'.xz', URL)

            # Construct RDF with POSCAR
            crystal = Structure.from_str(result.files['CONTCAR.relax.vasp'](), fmt='poscar')

            # Get elements in the compound
            elements = result.species
            last_element = elements[-1]
            last_element = last_element[:-1]
            elements[-1] = last_element

            # Collecting for sp_metals compound
            j = 0
            for element in elements:
                if element in sp_system:
                    j += 1
            if j == len(elements):
                X_sp_metals.append(RDF(crystal).RDF[1,:])
                dos = get_DOS_fermi(result.compound+'.txt', result)
                Y_sp_metals.append(dos)
                materials_info.append(material_properties(result, dos))

                print('progress: ', i+1, '/', n, '-------- material is stored')
            else:
                print('progress: ', i+1, '/', n, '-------- material is rejected')

        os.remove(result.compound+'.txt')

    except:
        print('progress: ', i+1, '/', n, '-------- material does not fit the criteria')
        os.remove(result.compound+'.txt')
        pass

# Save as json for sp metals
with open('sp_metal_aflow_844.json', 'w') as f:
    json.dump(materials_info, f, cls=NumpyEncoder, indent=1)

entry, E_form = read_json('sp_metal_aflow_844.json')
print(entry, E_form)
