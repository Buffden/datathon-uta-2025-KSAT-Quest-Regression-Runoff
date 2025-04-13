# f'$R^2$
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

# Load data
data = pd.read_csv(r'/Users/aminahmadisharaf/Downloads/behzad/amin/codes/USKSAT_OpenRefined_Cleaned.csv')

# Function to check if two values are not approximately equal
def isnotequal(x, y):
    if abs(x - y) < 10**-2:
        return False
    return True

# Identify invalid indices
invalidindecies = np.array(list(map(isnotequal, data.iloc[:, 11:16].sum(axis=1), data.iloc[:, 16])))
data = data[~invalidindecies]

# Drop rows with NaN in specified columns
columns_to_check = ['Ksat_cmhr', 'Db', 'OC', 'Clay', 'Silt', 'Sand', 'VCOS', 'COS', 'MS', 'FS', 'VFS', 'Depth.cm_Top', 'Depth.cm_Bottom', 'Dia.cm', 'Height.cm']
data = data.dropna(subset=columns_to_check)

# Convert columns to numeric
numeric_columns = ['Silt', 'Clay', 'Sand', 'Depth.cm_Top', 'Depth.cm_Bottom', 'Height.cm', 'Dia.cm', 'Db', 'VCOS', 'COS', 'MS', 'FS', 'VFS', 'Ksat_cmhr']
for col in numeric_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Remove unnecessary columns
data = data.drop(columns=['Ref', 'Site', 'Soil', 'Sand', 'Field', 'Method', 'Depth.cm_Bottom', 'Dia.cm', 'Height.cm'])

# Check for NaN values after conversions
is_NaN = data.isnull()
row_has_NaN = is_NaN.any(axis=1)
rows_with_NaN = data[row_has_NaN]
print(rows_with_NaN)

data['Ksat_cmhr'].describe()

# Create a figure and axis object
fig, ax = plt.subplots()

log_ks = np.log(data.Ksat_cmhr)
sns.histplot(log_ks, color='gray', ax=ax, label = 'Histogram of Ln(Ks)', bins=100)


# Set axis labels and title
ax.set_xlabel('Ln(Ks)', weight='bold')
ax.set_ylabel('Frequency', weight='bold')
# Add legend
ax.legend()

# Set tick label weights
ytick_labels = ax.get_yticklabels()  # Add this line
plt.setp(ytick_labels, weight='bold')
xtick_labels = ax.get_xticklabels()  # Add this line
plt.setp(xtick_labels, weight='bold')
# Show the plot


plt.savefig(r'/Users/aminahmadisharaf/Downloads/behzad/amin/codes/output/distribution_ksat_35.svg', format='svg')
plt.show()

