# Make a list of dicts of materials properties of ICSD materials with < 40 atoms per unit cell.

import os
import sys
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
def read_json(json_file):
    with open(json_file, "r") as f:
        content = json.load(f)
    entry = []
    E_form = []
    for dct in content:
        lattice = dct['lattice']
        coords = dct['coordinates']
        elements = dct['atom_array']
        E_form.append(dct['form_energy_cell'])
        entry.append(Structure(lattice, elements, coords))

    return entry, E_form

def material_properties(result):
    """

    """
    atoms = []
    for i, species in enumerate(result.species):
        for j in range(result.composition[i]):
            # Since AFLOW return value like ['Al', 'Si/n'], /n is needed to be gone!
            if len(species) > 2:
                atoms.append(species[:-2])
            else:
                atoms.append(species)
    lat = Lattice.from_lengths_and_angles(result.geometry[:3], result.geometry[3:])
    mat_property = {'catalog': result.catalog,
                    'formula': result.compound,
                    'lattice': lat.matrix,
                    'coordinates': result.positions_fractional,
                    'atom_array': atoms,
                    'form_energy_cell': result.enthalpy_formation_cell,
                    'n_atoms': result.natoms,
                    'volume': result.volume_cell,
                    'space_group': result.spacegroup_relax}
    return mat_property

####################################### Part a: Mining ###########################################
# Get materials from AFLOW database based on this given criteria:
results = search(catalog='icsd',batch_size = 100
         ).filter(K.natoms < 40)

n = len(results) # number of avaiable data points
materials_info = []
for i, result in enumerate(results):
    try:
        materials_info.append(material_properties(result))

        print('progress: ', i+1, '/', n, '-------- material is stored')

    except:
        print('progress: ', i+1, '/', n, '-------- material does not fit the criteria')
        pass

# Save as json for sp metals
with open('all_aflow.json', 'w') as f:
    json.dump(materials_info, f, cls=NumpyEncoder, indent=1)

#entry, E_form = read_json('all_aflow.json')
#print(entry, E_form)
