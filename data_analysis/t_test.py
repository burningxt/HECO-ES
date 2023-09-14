import numpy as np
import pandas as pd
from scipy.stats import ttest_ind_from_stats


__logBase10of2 = 3.010299956639811952137388947244930267681898814621085413104274611e-1


def RoundToSigFigs_fp(x, sigfigs):
    """
    Rounds the value(s) in x to the number of significant figures in sigfigs.
    Return value has the same type as x.

    Restrictions:
    sigfigs must be an integer type and store a positive value.
    x must be a real value or an array like object containing only real values.
    """
    if not (type(sigfigs) is int or type(sigfigs) is np.long or
            isinstance(sigfigs, np.integer)):
        raise TypeError("RoundToSigFigs_fp: sigfigs must be an integer.")
    if sigfigs <= 0:
        raise ValueError("RoundToSigFigs_fp: sigfigs must be positive.")
    if not np.all(np.isreal(x)):
        raise TypeError("RoundToSigFigs_fp: all x must be real.")
    # temporarily suppres floating point errors
    errhanddict = np.geterr()
    np.seterr(all="ignore")
    matrixflag = False
    if isinstance(x, np.matrix):  # Convert matrices to arrays
        matrixflag = True
        x = np.asarray(x)
    xsgn = np.sign(x)
    absx = xsgn * x
    mantissas, binaryExponents = np.frexp(absx)
    decimalExponents = __logBase10of2 * binaryExponents
    omags = np.floor(decimalExponents)
    mantissas *= 10.0 ** (decimalExponents - omags)
    if type(mantissas) is float or isinstance(mantissas, np.floating):
        if mantissas < 1.0:
            mantissas *= 10.0
            omags -= 1.0
    else:  # elif np.all(np.isreal( mantissas )):
        fixmsk = mantissas < 1.0,
        mantissas[fixmsk] *= 10.0
        omags[fixmsk] -= 1.0
    result = xsgn * np.around(mantissas, decimals=sigfigs - 1) * 10.0 ** omags
    if matrixflag:
        result = np.matrix(result, copy=False)
    np.seterr(**errhanddict)
    return result


for dimension in [10, 30, 50, 100]:
    df_HECO_MAES = pd.read_csv(f"table_{dimension}.csv")
    df_baselines = pd.read_csv(f"./baseline_data/baselines_{dimension}D.csv")
    # mean_MAES_2022 = df.iloc[3, 1:].to_numpy().astype(np.float)
    # std_MAES_2022 = df.iloc[5, 1:].to_numpy().astype(np.float)
    HECO_MAES = df_HECO_MAES.iloc[[3, 5, 7], 1:].to_numpy().T.astype(float)
    exclude = [16, 18, 25, 27]
    HECO_MAES = RoundToSigFigs_fp(np.delete(HECO_MAES, exclude, axis=0), 3)
    df = pd.DataFrame(
        {
            'HECO_MA-ES(mean)': HECO_MAES[:, 0],
            'HECO_MA-ES(std)': HECO_MAES[:, 1],
        }
    )
    baselines = df_baselines.iloc[:, :].to_numpy().astype(float)
    arr_pvalue = np.zeros(24)
    algorithms = ['vMA-ESbm', 'eMAgES', 'IUDE', 'CORCO', 'DeCODE']
    algorithm_id = 0
    for algorithm in algorithms:
        for problem_id in range(24):
            t_test_indResult = ttest_ind_from_stats(mean1=HECO_MAES[problem_id, 0], std1=HECO_MAES[problem_id, 1],
                                                    nobs1=25, mean2=baselines[problem_id, 3 * algorithm_id + 0],
                                                    std2=baselines[problem_id, 3 * algorithm_id + 1], nobs2=25,
                                                    alternative='two-sided')
            arr_pvalue[problem_id] = t_test_indResult.pvalue
        arr_compare = arr_pvalue < 0.05
        symbol_ = []
        for problem_id in range(24):
            if HECO_MAES[problem_id, 2] < baselines[problem_id, 3 * algorithm_id + 2]:
                symbol_.append('+')
            elif HECO_MAES[problem_id, 2] > baselines[problem_id, 3 * algorithm_id + 2]:
                symbol_.append('-')
            else:
                if arr_compare[problem_id]:
                    if HECO_MAES[problem_id, 0] < baselines[problem_id, 3 * algorithm_id + 0]:
                        if baselines[problem_id, 3 * algorithm_id + 0] > 1E-8:
                            symbol_.append('+')
                        else:
                            symbol_.append('=')
                    elif HECO_MAES[problem_id, 0] > baselines[problem_id, 3 * algorithm_id + 0]:
                        if HECO_MAES[problem_id, 0] > 1E-8:
                            symbol_.append('-')
                        else:
                            symbol_.append('=')
                else:
                    symbol_.append('=')

        df_ttest = pd.DataFrame(
            {
                f'{algorithms[algorithm_id]}(mean)': baselines[:, 3 * algorithm_id + 0],
                f'{algorithms[algorithm_id]}(std)': baselines[:, 3 * algorithm_id + 1],
                't': symbol_,
            }
        )
        for i in range(baselines.shape[1]):
            if baselines[i, 3 * algorithm_id + 2] > 0.0:
                df_ttest.iloc[i, 0] = '***'
                df_ttest.iloc[i, 1] = '***'
        df = pd.concat([df, df_ttest], axis=1)
        algorithm_id += 1
    df.to_csv(f't-test_{dimension}.csv')


