import pyiast
import pandas as pd
import numpy as np

#This script efficiently implements the pyIAST algorithm developed by Simon et al., 
#enabling batch processing of adsorption selectivity data. More info on pyIAST can be 
#found at https://pyiast.readthedocs.io/en/latest/ (as of 2022-10-10).

#The script takes an input (N + 1) x 8 csv, where each row is a unique sample with the necessary parameters
#to define its CO2 and N2 adsorption behaviors. The first row contains the column titles which are as follows:
#"Sample", "qA", "qB", "SA", "HA", "SB", "HB", "k_H". Sample references the name of the sample,
#qA and qB are the strong and weak site saturation capacities (mmol/g) derived from the CO2 isotherms
#fit by the daul site langmuir model, SA and SB are the assocatied entropies of adsorption (J/molK),
#HA and HB are the associated enthalpies of adsorption (kJ/mol), and k_H is the Henry's constant for N2
#in units of (mmol/g/Torr).

#The output is a unique .csv for each sample tabulating the predicted binary isotherm, purity, and selectivity.

#It is important to ensure that the column labels on the original input .csv follow above formatting exactly.

folder_path = "/Users/..."+"/" #make sure to change this to your absolute file path
file_name = "your-file-name-here.csv" #make sure to change the input file name here
params = pd.read_csv(folder_path + file_name)

YCO2_list = [.0004, .0006, .0008, .001, .002, .004, .008, .01, .02, .03, .04, .05, 
		.06, .07, .08, .09, .1, .15, .2, .4, .5, .6, .8, .99]

for index, row in params.iterrows():
    name = params['Sample'][index]
    qA = params['qA'][index]  # mmol/g
    delHA = params['HA'][index]  # kJ/mol
    delSA = params['SA'][index]  # J/molK
    qB = params['qB'][index]  # mmol/g
    delHB = params['HB'][index]  # kJ/mol
    delSB = params['SB'][index]  # mmol/g
    k_H = params['k_H'][index] # mmol/g/Torr

    # define bogus isotherms to overwrite later

    df_N2 = pd.DataFrame([[10000000, 1]], columns=["P", "N"])  # bogus data
    df_CO2 = pd.DataFrame([[10000000, 1]], columns=["P", "N"])  # bogus data

    N2_isotherm = pyiast.ModelIsotherm(df_N2, loading_key="N", pressure_key="P", model="Henry")
    CO2_isotherm = pyiast.ModelIsotherm(df_CO2, loading_key="N", pressure_key="P", model="DSLangmuir")

    # overwrite bogus isotherm parameters with real values from fitting

    N2_isotherm.params['KH'] = k_H
    CO2_isotherm.params['M1'] = qA
    CO2_isotherm.params['K1'] = np.exp(delSA / 8.314) * np.exp(
        -delHA * 1000 / 8.314 / 303.15) / 62.36 / 303.15  # bA @ 30C in 1/Torr
    CO2_isotherm.params['M2'] = qB
    CO2_isotherm.params['K2'] = np.exp(delSB / 8.314) * np.exp(
        -delHB * 1000 / 8.314 / 303.15) / 62.36 / 303.15  # bB @ 30C in 1/Torr

    qCO2_list = []
    qN2_list = []
    XCO2_list = []
    SIAST_list = []
    for YCO2 in YCO2_list:
        PCO2 = 760*YCO2
        PN2 = 760 - PCO2
        x_YCO2 = pyiast.iast([PCO2, PN2],[CO2_isotherm, N2_isotherm], verboseflag=False)
        qCO2 = x_YCO2[0]
        qN2 = x_YCO2[1]
        qCO2_list.append(qCO2)
        qN2_list.append(qN2)
        XCO2 = qCO2 / (qCO2 + qN2)
        XN2 = 1 - XCO2
        XCO2_list.append(XCO2)
        S = (XCO2/YCO2)/(XN2/(1-YCO2))
        SIAST_list.append(S)
    results = pd.DataFrame({"yCO2":YCO2_list, "qCO2":qCO2_list,"qN2":qN2_list, "XCO2":XCO2_list,"SIAST":SIAST_list})
    results.to_csv(folder_path + str(name) + ".csv")
