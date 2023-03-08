import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# plt.style.use('ggplot')
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 4))


x = ['10D', '30D', '50D', '100D']
arr_rank_mean = np.zeros((4, 10))
arr_rank_median = np.zeros((4, 10))
count = 0
for dim in [10, 30, 50, 100]:
    df_rank_mean = pd.read_csv(f'./rank_mean_{dim}D.csv')
    df_rank_median = pd.read_csv(f'./rank_median_{dim}D.csv')
    arr_rank_mean[count, :] = df_rank_mean.iloc[:, -1]
    arr_rank_median[count, :] = df_rank_median.iloc[:, -1]
    count += 1


# axs[0, 0].plot(x, y1, '*-', color='tomato', label='$e$MA-ES')
axs[0].plot(x, arr_rank_mean[:, 0], 'P-', color='darkgrey', label='CAL_LSAHDE')
axs[0].plot(x, arr_rank_mean[:, 1], '>-', color='orange', label='LSHADE44+IDE')
axs[0].plot(x, arr_rank_mean[:, 2], '<-', color='pink', label='LSAHDE44')
axs[0].plot(x, arr_rank_mean[:, 3], 's-', color='steelblue', label='LSAHDE_IEpsilon')
axs[0].plot(x, arr_rank_mean[:, 4], 'v-', color='thistle', label='UDE')
axs[0].plot(x, arr_rank_mean[:, 5], 'd-', color='black', label='ϵMA$g$ES')
axs[0].plot(x, arr_rank_mean[:, 6], '+-', color='mediumseagreen', label='IUDE')
# axs[0].plot(x, arr_rank_mean[:, 7], '^-', color='tan', label='DeCODE')
axs[0].plot(x, arr_rank_mean[:, 8], 'o-', color='blue', label='HECO-DE')
axs[0].plot(x, arr_rank_mean[:, 9], '*-', color='tomato', label='HECO-ES')



# axs[0].grid(True)
axs[0].set_xlabel('mean value')
axs[0].set_ylabel('rank values')
axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.8))

axs[1].plot(x, arr_rank_median[:, 0], 'P-', color='darkgrey', label='CAL_LSAHDE')
axs[1].plot(x, arr_rank_median[:, 1], '>-', color='orange', label='LSHADE44+IDE')
axs[1].plot(x, arr_rank_median[:, 2], '<-', color='pink', label='LSAHDE44')
axs[1].plot(x, arr_rank_median[:, 3], 's-', color='steelblue', label='LSAHDE_IEpsilon')
axs[1].plot(x, arr_rank_median[:, 4], 'v-', color='thistle', label='UDE')
axs[1].plot(x, arr_rank_median[:, 5], 'd-', color='black', label='ϵMA$g$ES')
axs[1].plot(x, arr_rank_median[:, 6], '+-', color='mediumseagreen', label='IUDE')
# axs[1].plot(x, arr_rank_median[:, 7], '^-', color='tan', label='DeCODE')
axs[1].plot(x, arr_rank_median[:, 8], 'o-', color='blue', label='HECO-DE')
axs[1].plot(x, arr_rank_median[:, 9], '*-', color='tomato', label='HECO-ES')
# axs[1].grid(True)
axs[1].set_xlabel('median solution')
axs[1].set_ylabel('rank values')
axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.8))



# plt.plot(x, y1, '*-', color='tomato', label='$e$MA-ES')
# plt.plot(x, y2, 'o-', color='darkgrey', label='$v$MA-ESbm')
# plt.plot(x, y3, 's-', color='steelblue', label='HECO-DE')
# plt.plot(x, y4, 'd-', color='tan', label='MA$g$ES')
# plt.plot(x, y5, 'v-', color='thistle', label='IUDE')
# plt.plot(x, y6, '+-', color='mediumseagreen', label='CROCO')
# plt.plot(x, y7, 'x-', color='orange', label='DeCODE')

fig.tight_layout()
plt.savefig('fig_ranks.eps', dpi=600)
plt.show()
