import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
import shap

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

#################################################################################
# Get the list of column names
columns = data.columns
import seaborn as sns

import seaborn as sns

# Calculate the number of rows and columns for subplots
num_rows = 4
num_cols = 3

# Create subplots layout
fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 10))
fig.tight_layout(pad=2.0)

# Loop through the columns and plot histograms
for i, column in enumerate(columns):
    row = i // num_cols
    col = i % num_cols
    
    ax = axes[row, col]
    sns.histplot(data[column], kde=False, ax=ax, bins = 35)
    #ax.set_title(column)
    ax.set_xlabel(column)
    ax.set_ylabel('')
    
    if i == 0:
        ax.set_xlabel('DT (cm)')
    elif i == 1:
        ax.set_xlabel(r'K$_{s}$ (cm/hr)')  # Modified title format
    elif i == 2:
        ax.set_xlabel('DB (g/cm$^3$)')  # Set X-label
        
    # Adjust x-axis limits for VCOS subplot
    
    # Manually change position for i = 9 or 10
    
    if i == 2:
            ax.set_xlim(0.5, 2)
    if i == 3:
        ax.set_xlim(0, 5)
        ax.set_xlabel("VCOS (%)")
        
    if i == 4:
        ax.set_xlim(0, 30)
        ax.set_xlabel("COS (%)")
        
    if i == 5:
        ax.set_xlabel("MS (%)")
        
    if i == 6:
                ax.set_xlim(0, 100)
                ax.set_xlabel("FS (%)")
    if i == 7:
        ax.set_xlabel("VFS (%)")

    if i == 8:
        ax.set_xlim(0, 40)
        ax.set_xlabel("Silt (%)")
    if i == 9:
            ax.set_xlim(0, 65)
            ax.set_xlabel("Clay (%)")
    if i == 10:
        ax.set_xlim(0, 10)
        ax.set_xlabel('OC (%)')


            


# Remove empty subplots for the last row
if num_cols > len(columns) % num_cols > 0:
    for j in range(len(columns) % num_cols, num_cols):
        axes[-1, j].remove()
# Add y-axis label in the middle of the figure
fig.text(0.00, 0.5, 'Frequency', va='center', rotation='vertical',fontsize=16)
#fig.text(0.5, -0.01, 'Values', va='center', rotation='horizontal',fontsize=16)

# Save the plot as an SVG file
save_path = r'/Users/aminahmadisharaf/Downloads/behzad/amin/codes/output/histogram_plots bin = 35.svg'
plt.savefig(save_path, format='svg', bbox_inches='tight')

# Show the plots

plt.show()
