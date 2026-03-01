BTSP v2.0 Date: 1/3/2026

Changes made by Biswaroop 

Updated LTCC.py module from the previous HH implementation of LTCC in excitable cardiac cells 
1. Ohmic Conductance model
2. Calcium dependent deactivation

Removed the Calcium dependent deactivation term as it was giving buggy results of very low LTCC values due to reduced consistency in logcal and external Ca levels.

To a simplified neuronal LTCC Model applying the GHZ implementation referenced from Mahajan et al.(2019)


#Explain the equations and the changes
