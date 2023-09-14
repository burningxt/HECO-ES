from src.heco_es import MAES
from benchmark.cec2017.cec2017 import load_mat
import timeit
import multiprocessing as mp
import pandas as pd


# def run(runs):
#     for dimension in [10, 30, 50, 100]:
#         for problem_id in range(1, 29):
#             o, m, m1, m2 = load_mat(problem_id, dimension)
#             results = MAES(problem_id, dimension, o, m, m1, m2).ma_es()
#             df_new = pd.DataFrame(
#                 {
#                     "obj": [results[0]],
#                     "vio": [results[1]],
#                     "c": [(results[2], results[3], results[4])],
#                 }
#             )
#             df_new.to_csv(f'./data_analysis/output_data/problem_{problem_id}_{dimension}D.csv',
#                           mode='a', index=False, header=False)
#
#
# if __name__ == '__main__':
#     __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
#     start = timeit.default_timer()
#     pool = mp.Pool(processes=25)
#     res = pool.map(run, range(25))
#     stop = timeit.default_timer()
#     print('Time: ', stop - start)


if __name__ == '__main__':
    start = timeit.default_timer()
    prob_id = 4
    dim = 100
    o, m, m1, m2 = load_mat(prob_id, dim)
    results = MAES(prob_id, dim, o, m, m1, m2).ma_es()
    df_data = results[-1]
    df_data.to_csv(f'./data_analysis/history_data.csv', index=False, header=False)
    stop = timeit.default_timer()
    print('Time: ', stop - start)


