import pandas as pd
import numpy as np


#The script takes an input (N + 1) x 8 csv, where each row is a unique sample with the necessary parameters
#to define its CO2 adsorption behavior. The first row contains the column titles which are as follows:
#"Sample", "qA", "qB", "SA", "HA", "SB", and "HB". Sample references the name of the sample,
#qA and qB are the strong and weak site saturation capacities (mmol/g) derived from the CO2 isotherms
#fit by the daul site langmuir model, SA and SB are the assocatied entropies of adsorption (J/molK), and
#HA and HB are the associated enthalpies of adsorption (kJ/mol). This script takes these inputs and produces 
#two outputs. The first output creates a csv for each sample containing desorption temperature, energy, working 
#capacity, and capacity per energy for three conditions of 4%, 10%, and 15% CO2 (1 bar). It also produces the 
#coordinates of the maximum of the working capacity per energy. The second output creates a  single csv containing 
#just the coordinates of the maximum working capacity per energy under the same three conditions for all samples.
#These are calculated by the simplified thermodynamic process analysis first proposed by Sculley et al. and 
#summarized in Eqs. 16-19 in the manuscript text.

#It is important to ensure that the column labels on the original input .csv follow above formatting exactly.

# constants
R = 8.314  # J/Kmol
R2 = 62.36  # torrL/molK
heatcapacity = 1
T_lst1 = np.arange(293, 323, 10)
P_lst = [30.4, 76, 114, 760]  # relevant pressures in torr
T_lst2 = np.arange(250, 705, 5)
T = np.append(T_lst2, 313)
T2 = np.sort(T)  # temp in K
T3 = (T2-273.15)  # temp in C
yCO2 = [0.04, 0.01, 0.15]  #relevant pressures


# import Data
folder_path = "/Users/..."+"/" #make sure to change this to fit the absolute path of your input file
file_name = "your-file-name-here.csv" #make sure to update your file name here 
params = pd.read_csv(folder_path + file_name)

# list initialization
maxx4 = []
maxy4 = []
maxx10 = []
maxy10 = []
maxx15 = []
maxy15 = []

# list
for index, row in params.iterrows():
    name = params['Sample'][index]
    qA = params['qA'][index]  # mmol/g
    HA = params['HA'][index]  # kJ/mol
    SA = params['SA'][index]  # J/molK
    qB = params['qB'][index]  # mmol/g
    HB = params['HB'][index]  # kJ/mol
    SB = params['SB'][index]  # mmol/g
    q1 = []
    q2 = []
    q3 = []
    q4 = []
    for temp in T2:
        bA2_temp = np.exp(SA / R) * np.exp(-HA * 1000 / R / temp) / R2 / temp
        bB2_temp = np.exp(SB / R) * np.exp(-HB * 1000 / R / temp) / R2 / temp
        for idx, item in enumerate(P_lst):
            if idx == 0:
                q1.append(qA * bA2_temp * P_lst[idx] / (1 + bA2_temp * P_lst[idx]) + qB * bB2_temp * P_lst[idx] / (
                        1 + bB2_temp * P_lst[idx]))
            if idx == 1:
                q2.append(qA * bA2_temp * P_lst[idx] / (1 + bA2_temp * P_lst[idx]) + qB * bB2_temp * P_lst[idx] / (
                        1 + bB2_temp * P_lst[idx]))
            if idx == 2:
                q3.append(qA * bA2_temp * P_lst[idx] / (1 + bA2_temp * P_lst[idx]) + qB * bB2_temp * P_lst[idx] / (
                        1 + bB2_temp * P_lst[idx]))
            if idx == 3:
                q4.append(qA * bA2_temp * P_lst[idx] / (1 + bA2_temp * P_lst[idx]) + qB * bB2_temp * P_lst[idx] / (
                        1 + bB2_temp * P_lst[idx]))
    q4array = np.array(q4)
    dfq1 = pd.DataFrame({"Temp": T2, "q1": q1, "q2": q2, "q3": q3, "q4": q4})
    q_ads = dfq1.loc[dfq1['Temp'] == (313)].iloc[0, :]  # q ads at 313K
    q_ads2 = q_ads[1:4]

    for idx, val in enumerate(q_ads2):
        if idx == 0:
            work_capacity1 = val - q4array
        if idx == 1:
            work_capacity2 = val - q4array
        if idx == 2:
            work_capacity3 = val - q4array

    T_lst4 = np.arange(273,343,10)
    bA_array = np.exp(SA / R) * np.exp(-HA * 1000 / R / T_lst4) / R2 / T_lst4
    bB_array = np.exp(SB / R) * np.exp(-HB * 1000 / R / T_lst4) / R2 / T_lst4

    alpha = (qA + qB - np.flip(q4array))[:, None] * (bA_array*bB_array) #helper variable
    beta = (qA - np.flip(q4array)[:, None]) * bA_array + (qB - np.flip(q4array)[:, None]) * bB_array

    P_mat = (-beta + (beta ** 2 + 4 * alpha * np.flip(q4array)[:, None]) ** 0.5)/2/alpha
    ln_P_mat = np.log(P_mat)
    m = np.polyfit(1/T_lst4, np.transpose(ln_P_mat), 1)[0]
    b = np.polyfit(1/T_lst4, np.transpose(ln_P_mat), 1)[1]

    Qst = m * -8.314/1000 #define units

    T5 = T2[::-1]
    qdes = q4array[::-1]
    hdes = [Qst[0]*qdes[0]]
    for idx,val in enumerate(Qst[1:]):
        hdes.append(hdes[idx]+ ((Qst[idx]+Qst[idx+1])/2)*(qdes[idx+1]-qdes[idx]))

    hads = []
    for qval in q_ads2:
        xhighidx = np.argmax(np.flip(q4array)[:,None] > qval)
        xlowidx = xhighidx - 1
        xhigh = np.flip(q4array)[:, None][xhighidx]
        xlow = np.flip(q4array)[:, None][xlowidx]
        hads.append(hdes[xhighidx] + (qval - xhigh) * (hdes[xlowidx] - hdes[xhighidx])/(xlow-xhigh))

#calculation of parameters

    hcp = T5*heatcapacity-313
    for idx,val in enumerate(hads):
        if idx == 0:
            delH1_full = val - np.array(hdes)
            qwork1_full = q_ads2[idx] - qdes
            energy1_full = val + hcp - hdes
            cap_per_energy_1_full = 1000 * qwork1_full / (energy1_full)
            discont_xpos1 = np.argmax(cap_per_energy_1_full)
            delH1 = delH1_full[:discont_xpos1]
            qwork1 = qwork1_full[:discont_xpos1]
            energy1 = energy1_full[:discont_xpos1]
            cap_per_energy_1 = cap_per_energy_1_full[:discont_xpos1]
            Xmaxcord1 = qwork1[np.argmax(cap_per_energy_1)]
            Ymaxcord1 = max(cap_per_energy_1)
            maxx4.append(Xmaxcord1)
            maxy4.append(Ymaxcord1)
            Temps1 = T5[:discont_xpos1]

        if idx == 1:
            delH2_full = val - np.array(hdes)
            qwork2_full = q_ads2[idx] - qdes
            energy2_full = val + hcp - hdes
            cap_per_energy_2_full = 1000 * qwork2_full / (energy2_full)
            discont_xpos2 = np.argmax(cap_per_energy_2_full)
            delH2 = delH2_full[:discont_xpos2]
            qwork2 = qwork2_full[:discont_xpos2]
            energy2 = energy2_full[:discont_xpos2]
            cap_per_energy_2 = cap_per_energy_2_full[:discont_xpos2]
            Xmaxcord2 = qwork2[np.argmax(cap_per_energy_2)]
            Ymaxcord2  = max(cap_per_energy_2)
            maxx10.append(Xmaxcord2)
            maxy10.append(Ymaxcord2)
            Temps2 = T5[:discont_xpos2]


        if idx == 2:
            delH3_full = val - np.array(hdes)
            qwork3_full = q_ads2[idx] - qdes
            energy3_full = val + hcp - hdes
            cap_per_energy_3_full = qwork3_full / (energy3_full / 1000)
            discont_xpos3 = np.argmax(cap_per_energy_3_full)
            delH3 = delH3_full[:discont_xpos3]
            qwork3 = qwork3_full[:discont_xpos3]
            energy3 = energy3_full[:discont_xpos3]
            cap_per_energy_3 = cap_per_energy_3_full[:discont_xpos3]
            Xmaxcord3 = qwork3[np.argmax(cap_per_energy_3)]
            Ymaxcord3 = max(cap_per_energy_3)
            maxx15.append(Xmaxcord3)
            maxy15.append(Ymaxcord3)
            Temps3 = T5[:discont_xpos3]




    maxcords = pd.DataFrame({"Xmaxcord0.04":Xmaxcord1, "Ymaxcord0.04":Ymaxcord1,"Xmaxcord0.1":Xmaxcord2, "Ymaxcord0.1":Ymaxcord2,"Xmaxcord0.15":Xmaxcord3, "Ymaxcord0.15":Ymaxcord3}, index = [0])
    results1 = pd.DataFrame({"Tdes0.04":Temps1,"energy0.04":energy1, "qwork0.04":qwork1,"cap_per_energy_0.04":cap_per_energy_1}) #, "energy0.1":energy2,"qwork0.1":qwork2,"cap_per_energy_0.1":cap_per_energy_2,"energy0.15":energy3, "qwork0.15":qwork3,"cap_per_energy_0.15":cap_per_energy_3})
    results2 = pd.DataFrame({"Tdes0.1":Temps2, "energy0.1":energy2,"qwork0.1":qwork2,"cap_per_energy_0.1":cap_per_energy_2})
    results3 = pd.DataFrame({"Tdes0.15":Temps3,"energy0.15":energy3, "qwork0.15":qwork3,"cap_per_energy_0.15":cap_per_energy_3})
    final = pd.concat([results1,results2, results3, maxcords], axis = 1)
    #final.to_csv(folder_path + str(name) + ".csv")

    qstdf = pd.DataFrame({'CO2load':qdes,"Qst":Qst})
    qstdf.to_csv(folder_path + 'Qst_'+ str(name) + ".csv")

corddf = pd.DataFrame({"Xmaxcord0.04":maxx4, "Ymaxcord0.04":maxy4,"Xmaxcord0.1":maxx10, "Ymaxcord0.1":maxy10,"Xmaxcord0.15":maxx15, "Ymaxcord0.15":maxy15})
finalcord = pd.concat([corddf,params['Sample']],axis = 1)
finalcord.to_csv(folder_path + 'cords' + ".csv")