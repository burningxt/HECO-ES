import numpy as np
import pandas as pd


for dimension in [10, 30, 50, 100]:
    df_table = pd.DataFrame(index=['Best', 'Median', '\overline{v}', 'mean', 'Worst', 'Std', 'FR', '\overline{vio}'],
                            columns=np.arange(1, 29))
    for problem_id in range(1, 29):
        df = pd.read_csv(f'./output_data/problem_{problem_id}_{dimension}D.csv')
        obj = df.iloc[:, 0]
        vio = df.iloc[:, 1]
        df_table.loc['Best', problem_id] = obj.min()
        df_table.loc['Median', problem_id] = obj.median()
        # df_table.loc['c', problem_id] = df.iloc[0, 2]
        df_table.loc['\overline{v}', problem_id] = vio.median()
        df_table.loc['mean', problem_id] = obj.mean()
        df_table.loc['Worst', problem_id] = obj.max()
        df_table.loc['Std', problem_id] = obj.std()
        df_table.loc['FR', problem_id] = 100 * vio[vio == 0.0].shape[0] / vio.shape[0]
        df_table.loc['\overline{vio}', problem_id] = vio.mean()
    df_table.to_csv(f"table_{dimension}.csv")
