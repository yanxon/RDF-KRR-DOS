# In this script, metallic materials is mined from Materials Project (MP) database.
# Structural and density of states (DOS) information is to be mined.
# From the structure, radial distribution function (RDF) can be constructed using the RDF code.
# DOS at fermi level can also be extracted.

# Importing libraries
import numpy as np
from pymatgen import MPRester
from RDF import *
from pymatgen.core.composition import Composition
from pymatgen.core.periodic_table import Element

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

#for data in datas:
#    formula.append(data['pretty_formula'])
#    ids.append(data['material_id'])
    
#points = len(ids)

sp_element = ["Be", "Li", "Na", "Mg", "Al", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga","Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn"]

for data in datas:
    compos = Composition(data['pretty_formula']).as_dict()
    compos = list(compos.keys())
    if len(compos) == 1:
        if Element[compos[0]].symbol in sp_element:
            formula.append(data['pretty_formula'] + ' ' + data['material_id'])
            ids.append(data['material_id'])
        else:
            pass
    elif len(compos) == 2:
        if Element[compos[0]].symbol in sp_element:
            if Element[compos[1]].symbol in sp_element:
                formula.append(data['pretty_formula'] + ' ' + data['material_id'])
                ids.append(data['material_id'])
            else:
                pass
        else:
            pass
    elif len(compos) == 3:
        if Element[compos[0]].symbol in sp_element:
            if Element[compos[1]].symbol in sp_element:
                if Element[compos[2]].symbol in sp_element:
                    formula.append(data['pretty_formula'] + ' ' + data['material_id'])
                    ids.append(data['material_id'])
                else:
                    pass
            else:
                pass
        else:
            pass

    else:
        pass


points = len(ids)

# Appending RDF and 
X = [] # Reserve for RDF structure; X is 1D array.
Y = [] # Reserve for DOS at fermi level; Y is 1D array.

for i, identity in enumerate(ids[:2]):
    try:
        # Mining DOS
        dos = m.get_dos_by_material_id(identity)
        energies = dos.energies - dos.efermi
        total_dos = sum(dos.densities.values()) # Sum both spins, if present
        
        combine = np.vstack((energies, total_dos))
        print(combine)
        # Extracting the DOS at fermi level.
        combine_abs = abs(combine[0,:])
        find_ele_at_fermi = np.where(combine_abs == min(combine_abs))
        ele_at_fermi = find_ele_at_fermi[0][0]

        # Constructing RDF
        structure = m.get_structure_by_material_id(identity)
        struc = RDF(structure)
        X.append(struc.RDF[1,:])
        Y.append(combine[1,ele_at_fermi])
        		
        print('Progress: ', i+1, '/', points, ' ---> ', formula[i])
        
    except:
        print('Progress: ', i+1, '/', points, ' is passed',  ' ---> ', formula[i])
        
np.savetxt('X_.txt', X)
np.savetxt('Y_.txt', Y)
