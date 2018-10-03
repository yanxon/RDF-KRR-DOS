# RDF ---> Density of States (DOS) at Fermi level.

A machine learning project to predict DOS at fermi level of sp and spd systems using RDF as descriptor. This project is inspired by:
https://journals.aps.org/prb/abstract/10.1103/PhysRevB.89.205118

Descriptor:
Radial distribution function (RDF)

Prediction:
DOS at Fermi level

## 1.    Create RDF code
   A RDF code (RDF.py) is created and tested with XCl (X = Li, Na, K) compounds.
   Check out:
   https://github.com/yanxon/RDF/blob/master/XCl_test.ipynb

## 2.    Machine learning procedure

   https://github.com/yanxon/RDF-DOS-KRR/blob/master/RDF_DOS_KRR.py (RDF ---> DOS, Aflow)
   
   This script extracts sp metals from AFLOW database. The sp metals are split into train and test datasets. Finally, KRR algorithm is used to train the train dataset and tested with the test dataset.

## 3.    Result
     
   Result (sp metals):
   
   ![krr_0 1](https://user-images.githubusercontent.com/32254481/46254622-d24fcc80-c446-11e8-8cf8-310630341efc.png)
   
   cap:  MAE = 0.0022 states/eV/A^3;
         r^2 = 0.6540

   Comparison to PRB (spd system):
   
   ![krr_prb_mp](https://user-images.githubusercontent.com/32254481/46270894-a98b0e00-c4fe-11e8-88d7-ac30552aecf9.png)
   
   cap:  MAE = 0.01258 states/eV/A^3;
         r^2 = 0.6748
