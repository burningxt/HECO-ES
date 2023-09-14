import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Adjusting the style
sns.set_palette("deep", 10)

# Load mean rank data
rank_mean_10D = pd.read_csv('./rank_mean_10D.csv')
rank_mean_30D = pd.read_csv('./rank_mean_30D.csv')
rank_mean_50D = pd.read_csv('./rank_mean_50D.csv')
rank_mean_100D = pd.read_csv('./rank_mean_100D.csv')

# Load median rank data
rank_median_10D = pd.read_csv('./rank_median_10D.csv')
rank_median_30D = pd.read_csv('./rank_median_30D.csv')
rank_median_50D = pd.read_csv('./rank_median_50D.csv')
rank_median_100D = pd.read_csv('./rank_median_100D.csv')

# Extract cumulative rank data for mean values
cumulative_rank_mean = {
    '10D': rank_mean_10D.iloc[:, -1].values,
    '30D': rank_mean_30D.iloc[:, -1].values,
    '50D': rank_mean_50D.iloc[:, -1].values,
    '100D': rank_mean_100D.iloc[:, -1].values
}

# Extract cumulative rank data for median values
cumulative_rank_median = {
    '10D': rank_median_10D.iloc[:, -1].values,
    '30D': rank_median_30D.iloc[:, -1].values,
    '50D': rank_median_50D.iloc[:, -1].values,
    '100D': rank_median_100D.iloc[:, -1].values
}

# Extract algorithm names
algorithms = rank_mean_10D.iloc[:, 0].values

# Different markers for clarity
# markers = ['o', 's', 'D', '^', 'v', 'p', '*', 'H', '<', '>']
markers = ['^', 'v', '<', 'o', '>', 's', 'D', '+', 'p', 'H', '*']

color = ['#4c72b0', '#dd8452', '#55a868', '#c44e52', '#8172b3', '#937860', '#da8bc3', '#8c8c8c', '#ccb974', '#64b5cd', '#6D8F3A']
# color = ['#4c72b0', '#dd8452', '#55a868', '#c44e52', '#8172b3', '#937860', '#da8bc3', '#8c8c8c', '#ccb974', '#64b5cd']

color_dict = {
    algorithms[0]: color[8],
    algorithms[1]: color[1],
    algorithms[2]: color[2],
    algorithms[3]: color[9],
    algorithms[4]: color[4],
    algorithms[5]: color[6],
    algorithms[6]: color[5],
    algorithms[7]: color[10],
    algorithms[8]: color[7],
    algorithms[9]: color[0],
    algorithms[10]: color[3],
}

# Dimensions
dimensions = ['10D', '30D', '50D', '100D']

# Create side-by-side subplots for both mean and median cumulative ranks
fig, axs = plt.subplots(1, 2, figsize=(14, 4))

# Plotting mean ranks
for idx, (algo, ranks) in enumerate(zip(algorithms, zip(*cumulative_rank_mean.values()))):
    color = color_dict.get(algo)
    axs[0].plot(dimensions, ranks, ':'+markers[idx], label=algo, color=color, linewidth=1.5, markersize=8)

axs[0].set_ylabel('Cumulative Rank (Mean)')
axs[0].set_xlabel('Dimension')
# axs[0].set_title('Mean values')
axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.6))

# Plotting median ranks
for idx, (algo, ranks) in enumerate(zip(algorithms, zip(*cumulative_rank_median.values()))):
    color = color_dict.get(algo)
    axs[1].plot(dimensions, ranks, ':'+markers[idx], label=algo, color=color, linewidth=1.5, markersize=8)

axs[1].set_ylabel('Cumulative Rank (Median)')
axs[1].set_xlabel('Dimension')
# axs[1].set_title('Median values')
axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.6))

# fig.suptitle('Cumulative Rank across Dimensions', fontsize=14)
plt.tight_layout()
plt.subplots_adjust(top=0.85)  # Adjusting space to avoid overlap with the main title
plt.savefig('fig_ranks.eps', dpi=1200)
plt.show()
