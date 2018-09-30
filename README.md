# RDF ---> Density of States (DOS) at Fermi level.

A machine learning project to predict DOS at fermi level of sp and spd systems using RDF as descriptor. This project is inspired by:
https://journals.aps.org/prb/abstract/10.1103/PhysRevB.89.205118

Descriptor:
Radial distribution function (RDF)

Prediction:
DOS at Fermi level

## 1.  Create RDF code (RDF.py)
   We made a RDF code (RDF.py) and test the validity with several XCl (X = Li, Na, K) compounds.
   Please check out:
   https://github.com/yanxon/RDF/blob/master/XCl_test.ipynb

## 2.  Two mining code to extract data from Aflow and Materials Project (MP) databases.

         a. https://github.com/yanxon/RDF/blob/master/get_RDF_DOS_from_AFLOW.py (RDF ---> DOS, Aflow)
            There are 3 types of metals: all metals, sp metals, and spd metals.
         b. https://github.com/yanxon/RDF/blob/master/MP_RDF_DOS.py (RDF ---> DOS, MP)

## 3.  Machine learning: Kernel Ridge Regression.
   
   Using 2a script, sp metals are extracted from AFLOW. Then, the sp metals is learned in KRR_AFLOW_sp_metals.py.
   
   Result:
   
   ![krr_0 1](https://user-images.githubusercontent.com/32254481/46254547-ee9f3980-c445-11e8-8c99-09da4e968b31.png)
   
   cap:  MAE = 0.0022 states/eV/A^3;
         r^2 = 0.6540
