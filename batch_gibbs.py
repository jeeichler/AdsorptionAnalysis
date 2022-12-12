# This code takes an input csv where the first column is the name of the samples and the following
# are the dual site langmuir fit parameters and there associated errors. A fixed coverage
# or loading is specified and then the pressure need to achieve that loading/coverage is
# determined. Then the van't hoff relation is used to extract the enthalpy and entropy of
# adsorption at any coverage. Uncertainty propagation was done at each step of arithmetic,
# and the uncertainty in the fit parameters was determined using a weighted lest squares
# regression.

import numpy as np
import pandas as pd
import statsmodels.api as sm

tlst = np.arange(313, 273,-10)
invtlst = 1/tlst
R = 8.314# J/Kmol
CO2_frac_covs = np.arange(.001,1,.005)# desired fraction coverages of CO2 to be solved for
#CO2_loading = np.arange(.01,3, .005)
sampledf = pd.DataFrame({"Sat Cap (%)":list(reversed(CO2_frac_covs))}) #data frame iniitalization
#sampledf = pd.DataFrame({"CO2loading ((mmol/g))":list(reversed(CO2_loading))})


# import Data
folder_path = "/Users/johneichler/Desktop/Thermo_props"+"/" # input path to a folder containing data
file_name = "Gibbs_Input.csv"  #file name
params = pd.read_csv(folder_path + file_name)

for index, row in params.iterrows():
    name = params['Sample'][index]
    qA = params['qa'][index]  # mmol/g
    qB = params['qb'][index]  # mmol/g
    qaerr = params['qaerr'][index]  # mmol/g
    qberr  = params['qberr'][index]  # mmol/g
    sat_q = qA + qB
    satqerr = np.sqrt(qaerr**2+qberr**2)


    bas = [params['bA40'][index],params['bA30'][index],params['bA20'][index],params['bA10'][index]]
    bbs = [params['bB40'][index],params['bB30'][index],params['bB20'][index],params['bB10'][index]]
    baerr = [params['bA40err'][index],params['bA30err'][index],params['bA20err'][index],params['bA10err'][index]]
    bberr = [params['bB40err'][index],params['bB30err'][index],params['bB20err'][index],params['bB10err'][index]]


    bavals = np.array(bas)
    baerrvals = np.array(baerr)
    bbvals = np.array(bbs)
    bberrvals = np.array(bberr)



    loading = sat_q*CO2_frac_covs
    #loading = CO2_loading
    #qa_loaderr = np.flip(np.sqrt(qaerr**2+(satqerr*CO2_frac_covs)**2))
    #qb_loaderr = np.flip(np.sqrt(qberr**2+(satqerr*CO2_frac_covs)**2))
    qa_loaderr = .0000000000000000000000001
    qb_loaderr = .0000000000000000000000001  #negligible error for a defined input


    ba_bb = bavals*bbvals
    ba_bberr = ba_bb*np.sqrt((baerrvals/bavals)**2 + (bberrvals/bbvals)**2)

    alpha = (qA +qB- np.flip(loading))[:,None] * (ba_bb)
    alphaerr = alpha*np.sqrt((satqerr/sat_q)**2+(ba_bberr/ba_bb)**2)

    beta1 = (qA-np.flip(loading))[:,None]*bavals
    beta1err = beta1*np.sqrt((qa_loaderr/(qA-np.flip(loading))**2)[:,None]+ (baerrvals/bavals)**2)

    beta2 = (qB-np.flip(loading))[:,None]*bbvals
    beta2err = beta2*np.sqrt((qb_loaderr/(qB-np.flip(loading))**2)[:,None]+ (bberrvals/bbvals)**2)

    beta = beta1+beta2
    betaerr = np.sqrt(beta1err**2+beta2err**2)

    sqr_beta_err = beta*beta*np.sqrt(2*(betaerr/beta)**2)

    gamma = 4*alpha*np.flip(loading)[:,None]
    #gammaerr = gamma*np.sqrt((alphaerr/alpha)**2+((((np.flip(CO2_frac_covs)*satqerr))/np.flip(loading))**2)[:,None])
    gammaerr = gamma * np.sqrt(
        (alphaerr / alpha) ** 2 + ((((np.flip(loading) * satqerr)) / np.flip(loading)) ** 2)[:, None])
    determinant = beta*beta+gamma
    determinanterr = np.sqrt(sqr_beta_err**2 + gammaerr**2)

    root = np.sqrt(determinant)
    rooterr = root*0.5*determinanterr/determinant


    num = -beta+root
    numerr = np.sqrt(betaerr**2+rooterr**2)

    pressure  = num/2/alpha
    pressureerr = pressure*np.sqrt((numerr/num)**2+(alphaerr/alpha)**2)

    pressure_bar = pressure*.0013322
    pressureerr_bar = pressureerr*.0013322

    ln_pbar = np.log(pressure_bar)
    ln_pbar_err = pressureerr_bar/pressure_bar

    invtlst = sm.add_constant(invtlst)

    mlst = []
    blst = []
    merr_lst = []
    berr_lst = []
    for idx,val in enumerate(ln_pbar):
        mod_wls = sm.WLS((val), invtlst, weights=1 / (ln_pbar_err[idx] ** 2))
        res = mod_wls.fit()
        mlst.append(res.params[1])
        blst.append(res.params[0])
        merr_lst.append(res.bse[1])
        berr_lst.append(res.bse[0])
    Hads_array = np.array(mlst)*8.314/1000
    Sads_array = -np.array(blst)*8.314
    Hads_err_array = np.array(merr_lst)*8.314/1000
    Sads_err_array = -np.array(berr_lst)*8.314
    Gads_array = Hads_array-303*(Sads_array)/1000
    Gads_err_array = Hads_err_array-303*(Sads_err_array)/1000


    sampledf["CO2loading (mmol/g)"]= list(reversed(loading))
    sampledf["Neg_Hads"] = -Hads_array
    sampledf["Neg_Hads_err"] = -Hads_err_array
    sampledf["Neg_Sads"] = -Sads_array
    sampledf["Neg_Sads_err"] = -Sads_err_array
    sampledf["Neg_Gads"] = -Gads_array
    sampledf["Neg_Gads_err"] = -Gads_err_array
    sampledf.to_csv(folder_path + name + ".csv")



