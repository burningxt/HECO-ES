import numpy as np
import pandas as pd
from scipy.stats import ttest_ind_from_stats, ttest_1samp


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


# Path to the extracted CSV files
extracted_csv_path = 'output_data/'  # Update this path if necessary

# Problems to exclude
exclude_problems = [17, 19, 26, 28]

for dimension in [10, 30, 50, 100]:
    # Load the baseline data
    df_baselines = pd.read_csv(f'./baseline_data/baselines_{dimension}D.csv')
    algorithms = df_baselines.columns[::3]  # Extracting algorithm names for the baseline means

    # Initialize the main DataFrame for the results
    df = pd.DataFrame()

    # Valid problem IDs (excluding specific problems)
    valid_problems = [i for i in range(1, 29) if i not in exclude_problems]

    # Loop over the valid problem IDs
    for problem_index, problem_id in enumerate(valid_problems):
        # Load and preprocess the data for each problem
        problem_file = f'problem_{problem_id}_{dimension}D.csv'
        problem_data_raw = pd.read_csv(extracted_csv_path + problem_file).iloc[:, 0].to_numpy()
        problem_data = RoundToSigFigs_fp(problem_data_raw, 3)

        # Perform the one-sample t-tests and symbol determination for each algorithm
        for algorithm_id, algorithm in enumerate(algorithms):
            baseline_mean = df_baselines.iloc[problem_index, algorithm_id * 3]  # Use problem_index
            baseline_std = df_baselines.iloc[problem_index, algorithm_id * 3 + 1]  # Use problem_index

            t_test_result = ttest_1samp(problem_data, baseline_mean)
            # Check variance of the data
            a = np.var(problem_data)
            b = problem_data.mean() == baseline_mean


            # Determine the symbol based on comparison logic
            if df_baselines.iloc[problem_index, algorithm_id * 3 + 2] > 0.0:
                symbol = '+'
            else:
                if t_test_result.pvalue < 0.05 and problem_data.mean() < baseline_mean and abs(baseline_mean) > 1E-8:
                    symbol = '+'
                elif t_test_result.pvalue < 0.05 and problem_data.mean() > baseline_mean and abs(problem_data.mean()) > 1E-8:
                    symbol = '-'
                else:
                    symbol = '='
            if np.var(problem_data) <= 1e-8 and abs(problem_data.mean() - baseline_mean) < 1e-8:
                symbol = '='
            # Add the results to the DataFrame
            df.loc[problem_id, f'{algorithm}(mean)'] = baseline_mean
            df.loc[problem_id, f'{algorithm}(std)'] = baseline_std
            df.loc[problem_id, f'{algorithm}(t)'] = symbol

            # Apply additional conditions to the DataFrame
            if df_baselines.iloc[problem_index, algorithm_id * 3 + 2] > 0.0:
                df.loc[problem_id, f'{algorithm}(mean)'] = '***'
                df.loc[problem_id, f'{algorithm}(std)'] = '***'

    # Export to CSV
    df.to_csv(f't-test_{dimension}.csv')
