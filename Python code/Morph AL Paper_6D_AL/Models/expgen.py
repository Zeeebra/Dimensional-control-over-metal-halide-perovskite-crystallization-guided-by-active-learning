# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 16:42:56 2020

@author: Zhi Li
"""

import os, fnmatch
import matplotlib as mpl
import math
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus


def robot_file_gen (data, filename, R_2_A = 2.32, R_2_B = 2.91, R_3 = 2.36, tot_vol = 250):
    conc = np.array((data.iloc[:,:3]))
    V2 = (tot_vol*conc[:,0])/R_2_A # reagent 2 volume
    V3 = (tot_vol*conc[:,1]-V2*R_2_B)/R_3 # reagent 3 volume
    VFAH = ((tot_vol*conc[:,2])/1000)*46/1.22 # reagent 4 volume
    V1 = tot_vol-V2-V3-VFAH # reagent 1 volume

    V1 = V1.round()
    V2 = V2.round()
    V3 = V3.round()
    VFAH = VFAH.round()
    
    # Generate new robot input files
    df_rbtinput=pd.read_excel('Models/RobotInput_template.xls')

    pipette_low = "Tip_50ul_Water_DispenseJet_Empty"
    pipette_med = "StandardVolume_Water_DispenseJet_Empty"
    pipette_high = "HighVolume_Water_DispenseJet_Empty"

    df_rbtinput['Reagent1 (ul)'] = V1
    df_rbtinput['Reagent2 (ul)'] = V2
    df_rbtinput['Reagent3 (ul)'] = V3
    df_rbtinput['Reagent8 (ul)'] = VFAH
    df_rbtinput['Parameter Values'][0] = 80 # reaction temperature
    df_rbtinput['Parameter Values'][1] = 500 # stir rate
    df_rbtinput['Parameter Values'][2] = 900 # mixing time 1
    df_rbtinput['Parameter Values'][3] = 1200 # mixing time 2
    df_rbtinput['Parameter Values'][4] = 57600 # reaction time
    df_rbtinput['Parameter Values'][5] = 25 # Preheat temperature

    def liquid_class (vol):
        if max(vol) <= 50:
            output = pipette_low
        elif (max(vol) > 50) & (max(vol) <= 300):
            output = pipette_med
        elif (max(vol) > 300) & (max(vol) <= 1000):
            output = pipette_high
        else:
            raise ValueError("ValueError of volume given")

        return output

    df_rbtinput['Liquid Class'][0] = liquid_class(df_rbtinput['Reagent1 (ul)'])
    df_rbtinput['Liquid Class'][1] = liquid_class(df_rbtinput['Reagent2 (ul)'])
    df_rbtinput['Liquid Class'][2] = liquid_class(df_rbtinput['Reagent3 (ul)'])
    df_rbtinput['Liquid Class'][3] = liquid_class(df_rbtinput['Reagent4 (ul)'])
    df_rbtinput['Liquid Class'][4] = liquid_class(df_rbtinput['Reagent5 (ul)'])
    df_rbtinput['Liquid Class'][5] = liquid_class(df_rbtinput['Reagent6 (ul)'])
    df_rbtinput['Liquid Class'][6] = liquid_class(df_rbtinput['Reagent7 (ul)'])
    df_rbtinput['Liquid Class'][7] = liquid_class(df_rbtinput['Reagent8 (ul)'])
    df_rbtinput['Liquid Class'][8] = liquid_class(df_rbtinput['Reagent9 (ul)'])

    df_rbtinput.to_csv("Data/"+filename+"_robotinput.csv")
    
def conc_to_vol (conc):
    
    # Below are concentrations for each reagent: _a is PbI2, _b is morph, _c is solvent vol fraction (e.g. DMSO / DMSO solution)
    # R1: DMF, R2:DMSO, R3:GBL, R4:morph/Pb in DMF, R5:morph in DMF, R6:morph/Pb in DMSO, R7:morph in DMSO, 
    # R8:morph/Pb in GBL, R9:morph in GBL, R10:FAH, R11:H2O, R12:DCM (it is separate vial, not taken account for total solution volume)
    # Below are concentrations for each reagent: _a is PbI2, _b is morph, _c is solvent vol fraction (e.g. DMSO / DMSO solution)
    #c4_a = 2.32
    #c4_b = 2.91
    #c4_c = 0.494
    ############
    #c5 = 2.36
    #c5_c = 0.731
    ############
    #c6_a = 1.64
    #c6_b = 1.64
    #c6_c = 0.691
    ############
    #c7 = 4.7
    #c7_c = 0.467
    ############
    #c8_a = 0.71
    #c8_b = 0.35
    #c8_c = 0.907
    ############
    #c9 = 0.99
    #c9_c = 0.892
    ################

    # x is volume list

    def eq1(x):
        return (x[3]*2.32 + x[5]*1.64 + x[7]*0.71)/300 - conc['Pb']

    def eq2(x):
        return (x[3]*2.91 + x[5]*1.64 + x[7]*0.35 + x[4]*2.36 + x[6]*4.7 + x[8]*0.99)/300 - conc['morph']

    def eq3(x):
        return (x[1] + x[5]*0.691 + x[6]*0.467)/(x[0] + x[1] + x[2] \
                                                 + x[3]*0.494 + x[4]*0.731 + x[5]*0.691 \
                                                 + x[6]*0.467 + x[7]*0.907 + x[8]*0.892) - conc['DMSO']
    def eq4(x):
        return (x[2] + x[7]*0.907 + x[8]*0.892)/(x[0] + x[1] + x[2] \
                                                 + x[3]*0.494 + x[4]*0.731 + x[5]*0.691 \
                                                 + x[6]*0.467 + x[7]*0.907 + x[8]*0.892) - conc['GBL']

    def eq5(x):
        return (x[9]*1.22/46)/(300/1000) - conc['FAH']

    def eq6(x):
        return (x[10]*0.998/18)/(300/1000) - conc['H2O']

    def eq7(x):
        return sum(x)-300

    def obj(x):
        return -(x[0]+x[1]+x[2])

    b = (0,300)
    bnds = (b,b,b,b,b,b,b,b,b,b,b)

    cons = [{'type': 'eq', 'fun' : eq1}, \
            {'type': 'eq', 'fun' : eq2}, \
            {'type': 'eq', 'fun' : eq3}, \
            {'type': 'eq', 'fun' : eq4}, \
            {'type': 'eq', 'fun' : eq5}, \
            {'type': 'eq', 'fun' : eq6}, \
            {'type': 'eq', 'fun' : eq7}]

    x0 = np.array([27]*11)
    res = minimize(obj, x0, constraints=cons, bounds = bnds, method = 'SLSQP')
    vol = res.x
    
    return vol

def robot_file_gen_R8 (data, filename):
    
    df_rbtinput=pd.read_excel('Models/RobotInput_template_8R.xls')
    pipette_low = "Tip_50ul_Water_DispenseJet_Empty"
    pipette_med = "StandardVolume_Water_DispenseJet_Empty"
    pipette_high = "HighVolume_Water_DispenseJet_Empty"
    
    df_rbtinput['Index'] = list(data.index)
    df_rbtinput['Reagent1 (ul)'] = np.array(data['R1']) # pure DMF
    df_rbtinput['Reagent2 (ul)'] = np.array(data['R2']) # pure DMSO
    df_rbtinput['Reagent3 (ul)'] = np.array(data['R3']) # pure GBL
    df_rbtinput['Reagent4 (ul)'] = np.array(data['R4']) # Pb/morph DMF
    df_rbtinput['Reagent5 (ul)'] = np.array(data['R5']) # morph DMF
    df_rbtinput['Reagent7 (ul)'] = np.array(data['R6']) # formic acid
    df_rbtinput['Reagent8 (ul)'] = np.array(data['R7']) # water


    def liquid_class (vol):
        if max(vol) <= 50:
            output = pipette_low
        elif (max(vol) > 50) & (max(vol) <= 300):
            output = pipette_med
        elif (max(vol) > 300) & (max(vol) <= 1000):
            output = pipette_high
        else:
            raise ValueError("ValueError: volume given is out of range")
        return output

    df_rbtinput['Liquid Class'][0] = liquid_class(df_rbtinput['Reagent1 (ul)'])
    df_rbtinput['Liquid Class'][1] = liquid_class(df_rbtinput['Reagent2 (ul)'])
    df_rbtinput['Liquid Class'][2] = liquid_class(df_rbtinput['Reagent3 (ul)'])
    df_rbtinput['Liquid Class'][3] = liquid_class(df_rbtinput['Reagent4 (ul)'])
    df_rbtinput['Liquid Class'][4] = liquid_class(df_rbtinput['Reagent5 (ul)'])
    df_rbtinput['Liquid Class'][5] = liquid_class(df_rbtinput['Reagent6 (ul)'])
    df_rbtinput['Liquid Class'][6] = liquid_class(df_rbtinput['Reagent7 (ul)'])
    df_rbtinput['Liquid Class'][7] = liquid_class(df_rbtinput['Reagent8 (ul)'])
    df_rbtinput['Liquid Class'][8] = liquid_class(df_rbtinput['Reagent9 (ul)'])

    df_rbtinput.to_csv(filename+'.csv')