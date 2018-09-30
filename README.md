# RDF ---> Density of States (DOS) at Fermi level.

A machine learning project to predict DOS at fermi level of sp and spd systems using RDF as descriptor. This project is inspired by:
https://journals.aps.org/prb/abstract/10.1103/PhysRevB.89.205118

Descriptor:
Radial distribution function (RDF)

Prediction:
DOS at Fermi level

# 1.  Create RDF code (RDF.py)
   We made a RDF code (RDF.py) and test the validity using several XCl (X = Li, Na, K) compounds.
   Please check out:
   https://github.com/yanxon/RDF/blob/master/XCl_test.ipynb

# 2.  Two mining code to extract data from Aflow and Materials Project (MP) databases:

         a. https://github.com/yanxon/RDF/blob/master/get_RDF_DOS_from%20AFLOW.py (RDF ---> DOS, Aflow)
         b. https://github.com/yanxon/RDF/blob/master/MP_RDF_DOS.py (RDF ---> DOS, MP)

# 3.  Machine learning:
