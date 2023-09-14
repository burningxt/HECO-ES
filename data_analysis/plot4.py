import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
new_data = pd.read_csv('./history_data_old.csv')

# Rename columns
new_data.columns = ["fes", "weight_e", "weight_v", "best_f", "best_v", "δ_e", "δ_v", "case"]

# # Pre-process the data
# new_data['δ_e'] = new_data['δ_e'].str.replace('[', '').str.replace(']', '').astype(float)
# new_data['δ_v'] = new_data['δ_v'].str.replace('[', '').str.replace(']', '').astype(float)

# Set the color palette
palette = sns.color_palette("tab10", 5)

fig, ax1 = plt.subplots(figsize=(6, 3.5))

# Plot "best_f" and "best_v" with solid lines
ax1.plot(new_data['fes'], new_data['best_f'], label='$f_{best}$', color=palette[0], linestyle='-')
# ax1.plot(new_data['fes'], new_data['best_e'], label='$e_{best}$', color=palette[0], linestyle='-')
ax1.plot(new_data['fes'], new_data['best_v'], label='$v_{best}$', color=palette[3], linestyle='-')
ax1.set_xlabel('$FEs$')
ax1.set_ylabel('$f_{best}$ & $v_{best}$')
ax1.legend(loc='upper left')

# Create a second y-axis for "weight_e" and "weight_v", and plot them with dashed lines
ax2 = ax1.twinx()
ax2.plot(new_data['fes'], new_data['weight_e'], label='$\overline{w}_e$', linestyle='--', linewidth=0.6, color=palette[2])
ax2.plot(new_data['fes'], new_data['weight_v'], label='$\overline{w}_v$', linestyle='--', linewidth=0.6, color=palette[1])
ax2.set_ylabel('$\overline{w}_e$ & $\overline{w}_v$')
ax2.legend(loc='upper right')

# Mark "case" as Longitudinal reference lines wherever it's not equal to 0
for index, row in new_data.iterrows():
    if row['case'] != 0:
        ax1.axvline(x=row['fes'], color='gray', linestyle='-', linewidth=0.5)
        # ax1.text(row['fes'], ax1.get_ylim()[1], str(row['case']), rotation=90, verticalalignment='top', color='gray')

# Set the title and display the plot
# ax1.set_title('Variables vs. fes with Updated Line Styles')
plt.savefig('fig_adaptive.eps', dpi=1200)
plt.show()
