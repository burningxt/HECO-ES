import numpy as np
import pandas as pd


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

def rank(arr, n_alg):
    rank_value = np.full((24 + 1, n_alg), 1.0)
    for i in range(24):
        for j in range(n_alg):
            # for k in range(arr.shape[0]):
            for k in range(j + 1, n_alg):
                if arr.shape[2] == 3:
                    if arr[i, j, 2] > arr[i, k, 2]:
                        rank_value[i, j] += 1
                    elif arr[i, j, 2] < arr[i, k, 2]:
                        rank_value[i, k] += 1
                    else:
                        if arr[i, j, 1] > arr[i, k, 1]:
                            rank_value[i, j] += 1
                        elif arr[i, j, 1] < arr[i, k, 1]:
                            rank_value[i, k] += 1
                        else:
                            if arr[i, j, 0] > arr[i, k, 0]:
                                rank_value[i, j] += 1
                            elif arr[i, j, 0] < arr[i, k, 0]:
                                rank_value[i, k] += 1
                elif arr.shape[2] == 2:
                    if arr[i, j, 1] > arr[i, k, 1]:
                        rank_value[i, j] += 1
                    elif arr[i, j, 1] < arr[i, k, 1]:
                        rank_value[i, k] += 1
                    else:
                        if arr[i, j, 0] > arr[i, k, 0]:
                            rank_value[i, j] += 1
                        elif arr[i, j, 0] < arr[i, k, 0]:
                            rank_value[i, k] += 1
    return rank_value



algorithms = ['CAL_LSAHDE', 'LSHADE44+IDE', 'LSAHDE44', 'UDE', 'eMA-ES', 'HECO-DE', '$\epsilon$MA$g$ES', 'IUDE',
              'LSAHDE_IEpsilon', 'DeCODE', 'CORCO']
n_alg = len(algorithms)
for dim in [10, 30, 50, 100]:
    arr_mean = np.zeros((24, len(algorithms), 3))
    arr_median = np.zeros((24, len(algorithms), 2))
    df_FR = pd.read_csv(f'./baseline_data/cec2018/FR_{dim}D.csv')
    df_mean_obj = pd.read_csv(f'./baseline_data/cec2018/mean_obj_{dim}D.csv')
    df_mean_vio = pd.read_csv(f'./baseline_data/cec2018/mean_vio_{dim}D.csv')
    df_median_obj = pd.read_csv(f'./baseline_data/cec2018/median_obj_{dim}D.csv')
    df_median_vio = pd.read_csv(f'./baseline_data/cec2018/median_vio_{dim}D.csv')
    exclude = []
    arr_mean[:, :, 0] = RoundToSigFigs_fp(np.delete(df_mean_obj.iloc[:, 1:].to_numpy().astype(float), exclude, axis=0), 3)
    arr_mean[:, :, 1] = RoundToSigFigs_fp(np.delete(df_mean_vio.iloc[:, 1:].to_numpy().astype(float), exclude, axis=0), 3)
    arr_mean[:, :, 2] = RoundToSigFigs_fp(-np.delete(df_FR.iloc[:, 1:].to_numpy().astype(float), exclude, axis=0), 3)
    arr_mean[abs(arr_mean) < 1E-10] = 0.0
    arr_median[:, :, 0] = RoundToSigFigs_fp(np.delete(df_median_obj.iloc[:, 1:].to_numpy().astype(float), exclude, axis=0), 3)
    arr_median[:, :, 1] = RoundToSigFigs_fp(np.delete(df_median_vio.iloc[:, 1:].to_numpy().astype(float), exclude, axis=0), 3)
    arr_median[abs(arr_median) < 1E-10] = 0.0
    rank_mean_value = rank(arr_mean, n_alg).astype(int)
    rank_median_value = rank(arr_median, n_alg).astype(int)
    rank_mean_value[-1, :] = rank_mean_value.sum(axis=0) - 1
    rank_median_value[-1, :] = rank_median_value.sum(axis=0) - 1
    df_mean_rank = pd.DataFrame(
        {
            'CAL_LSAHDE': rank_mean_value[:, 0],
            'LSHADE44+IDE': rank_mean_value[:, 1],
            'LSAHDE44': rank_mean_value[:, 2],
            'UDE': rank_mean_value[:, 3],
            'LSAHDE_IEpsilon': rank_mean_value[:, 8],
            '$\epsilon$MA$g$ES': rank_mean_value[:, 6],
            'IUDE': rank_mean_value[:, 7],
            'CORCO': rank_median_value[:, 10],
            'DeCODE': rank_median_value[:, 9],
            'HECO-DE': rank_mean_value[:, 5],
            'HECO-ES': rank_mean_value[:, 4],
        }
    )
    df_median_rank = pd.DataFrame(
        {
            'CAL_LSAHDE': rank_median_value[:, 0],
            'LSHADE44+IDE': rank_median_value[:, 1],
            'LSAHDE44': rank_median_value[:, 2],
            'UDE': rank_median_value[:, 3],
            'LSAHDE_IEpsilon': rank_median_value[:, 8],
            '$\epsilon$MA$g$ES': rank_median_value[:, 6],
            'IUDE': rank_median_value[:, 7],
            'CORCO': rank_median_value[:, 10],
            'DeCODE': rank_median_value[:, 9],
            'HECO-DE': rank_median_value[:, 5],
            'HECO-ES': rank_median_value[:, 4],
        }
    )
    df_mean_rank.transpose().to_csv(f'rank_mean_{dim}D.csv')
    df_median_rank.transpose().to_csv(f'rank_median_{dim}D.csv')
    a = 0