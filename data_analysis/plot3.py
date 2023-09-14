import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the sensitivity analysis data
sensitivity_data = pd.read_excel("sensitive_data.xlsx")

# Setting Seaborn style and palette
# sns.set_style("whitegrid")
sns.set_palette("deep")

# Extract data for plotting
theta_values = [0.1, 0.15, 0.2, 0.25, 0.3]
c06_values = sensitivity_data.iloc[1, 1::2].astype(int).tolist()
c11_values = sensitivity_data.iloc[1, 2::2].astype(int).tolist()

# Plotting with Seaborn
plt.figure(figsize=(5,3))
sns.lineplot(x=theta_values, y=c06_values, marker='p', label='C06', linewidth=0, markersize=10)
sns.lineplot(x=theta_values, y=c11_values, marker='^', label='C11', linewidth=0, markersize=10)

# Setting labels, title, and legend
plt.xlabel('$θ$')
plt.ylabel('Feasible Rate (%)')
# plt.title('Sensitivity Analysis for C06 and C11')
plt.legend()
plt.tight_layout()
plt.savefig('fig_sensitive.eps', dpi=1200)
plt.show()
