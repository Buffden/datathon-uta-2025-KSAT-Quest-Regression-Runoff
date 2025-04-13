# f'$R^2$
import numpy as np
import pandas as pd
import sys
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from matplotlib import rc, rcParams
import matplotlib.patches as mpl_patches
import math
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from scipy.stats import randint, uniform

#from google.colab import drive
#from google.colab import files
#drive.mount("/content/drive", force_remount=True)
#%cd /content/drive/My Drive/Pachevsky_sadra

"""## Read data"""
data = pd.read_csv(r'/Users/aminahmadisharaf/Downloads/behzad/amin/codes/USKSAT_OpenRefined_Cleaned.csv')



def isnotequal (x,y):
  if abs(x-y)<10**-2:
    return False
  return True

invalidindecies = np.array(list(map(isnotequal, data.iloc[: , 11:16].sum(axis=1), data.iloc[: , 16])))
data = data[~invalidindecies]
data.columns

"""## Drop rows with nan cell"""

data = data.dropna(subset=['Ksat_cmhr'])
data = data.dropna(subset=['Db'])
data = data.dropna(subset=['OC'])
data = data.dropna(subset=['Clay'])
data = data.dropna(subset=['Silt'])
data = data.dropna(subset=['Sand'])
data = data.dropna(subset=['VCOS'])
data = data.dropna(subset=['COS'])
data = data.dropna(subset=['MS'])
data = data.dropna(subset=['FS'])
data = data.dropna(subset=['VFS'])
data = data.dropna(subset=['Depth.cm_Top'])
data = data.dropna(subset=['Depth.cm_Bottom'])
data = data.dropna(subset=['Dia.cm'])
data = data.dropna(subset=['Height.cm'])
data

"""## Convert to Numeric"""

data['Silt'] = pd.to_numeric(data['Silt'], errors='coerce')
data['Clay'] = pd.to_numeric(data['Clay'], errors='coerce')
data['Sand'] = pd.to_numeric(data['Sand'], errors='coerce')
data['Depth.cm_Top'] = pd.to_numeric(data['Depth.cm_Top'], errors='coerce')
data['Depth.cm_Bottom'] = pd.to_numeric(data['Depth.cm_Bottom'], errors='coerce')
data['Height.cm'] = pd.to_numeric(data['Height.cm'], errors='coerce')
data['Dia.cm'] = pd.to_numeric(data['Dia.cm'], errors='coerce')
data['Db'] = pd.to_numeric(data['Db'], errors='coerce')
data['VCOS'] = pd.to_numeric(data['VCOS'], errors='coerce')
data['COS'] = pd.to_numeric(data['COS'], errors='coerce')
data['MS'] = pd.to_numeric(data['MS'], errors='coerce')
data['FS'] = pd.to_numeric(data['FS'], errors='coerce')
data['VFS'] = pd.to_numeric(data['VFS'], errors='coerce')
data['Ksat_cmhr'] = pd.to_numeric(data['Ksat_cmhr'], errors='coerce')

"""## Correlation numeric show"""


# Calculate the Pearson correlation coefficients for all variables



data = data.drop(columns=['Ref',
                          'Site',
                          'Soil',
                          'Sand',
                          'Field',
                          'Method',
                          'Depth.cm_Bottom',
                          'Dia.cm',
                          'Height.cm'])

"""## Check list of data for possible Nan cells"""

is_NaN = data.isnull()
row_has_NaN = is_NaN.any(axis=1)
rows_with_NaN = data[row_has_NaN]

print(rows_with_NaN)

y = np.log(data.loc[:, 'Ksat_cmhr'])
print(y.min())

"""## Select Features and target"""

param_df = pd.DataFrame(columns=['iteration', 'n_estimators', 'min_child_weight', 'gamma', 'learning_rate', 
                              'subsample', 'colsample_bytree', 'max_depth', 'objective', 'booster'])

results = pd.DataFrame()

r2_aggregated_TestScore =[]
r2_aggregated_TrainScore =[]
bunch_sample = np.arange(2000, len(data), 2000)
bunch_sample= np.append(bunch_sample,len(data))
fig, axs = plt.subplots(3, 3, figsize=(15, 15))
Group = ['a','b','c','d','e','f','g','h','i']
#Group = ["ABCDEFGHI"]
iterator= [0,1,2,3,4,5,6,7,8,9]
for i, bunch, group in zip(iterator,bunch_sample,Group):
    X = data.loc[:, data.columns != 'Ksat_cmhr']
    y = np.log(data.loc[:, 'Ksat_cmhr'])
# Randomly pick bunch of data by 2000 step
    slection = bunch / len(X)
    if slection != 1:
        X, X_test, y, y_test = train_test_split(X,y,test_size=1- slection,random_state=1)
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=1)
        print(len(X_train))
    else:
        
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=1)
        print(len(X_train))

    
    """## Feature Scaling"""
    
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train.values[:, :] = sc.fit_transform(X_train.values[:, :])
    X_test.values[:, :] =  sc.transform(X_test.values[:, :])
    
    
    
    
    """## XGBoost"""
    
    # model tuning
    
    from xgboost import XGBRegressor
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import RandomizedSearchCV
    import time
    
    # A parameter grid for XGBoost
    params = {
        'n_estimators': randint(100, 500),
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.01, 0.3),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'gamma': uniform(0, 5),
        'min_child_weight': randint(1, 10),
        'reg_alpha': uniform(0, 1),
        'reg_lambda': uniform(0, 1)}
    
    reg = XGBRegressor(objective='reg:squarederror', nthread=4)
    
    # run randomized search
   # run randomized search
    n_iter_search = 100
    random_search = RandomizedSearchCV(reg,
                                      param_distributions=params,
                                      n_iter=n_iter_search,
                                      cv=5,
                                      scoring='neg_mean_squared_error')
   
    start = time.time()
    random_search.fit(X_train, y_train)

    best_params = random_search.best_params_
    best_regressor = random_search.best_estimator_
    random_search.best_params_
    
    """## Get predictions"""
    
    y_pred_test  = best_regressor.predict(X_test)
    
    
    y_test_flat = y_test.values.flatten()  # Converts DataFrame column to 1D array

    # Create a temporary DataFrame to hold this iteration's results
    temp_df = pd.DataFrame({
        f'y_pred_test_{group}': y_pred_test,
        f'y_test_{group}': y_test_flat
    })
    
    # If it's the first iteration, initialize results with temp_df
    if results.empty:
        results = temp_df
    else:
        # Join the new predictions and actuals to the main DataFrame
        results = pd.concat([results, temp_df], axis=1)

# Save the DataFrame to CSV
file_path = (r'/Users/aminahmadisharaf/Downloads/behzad/amin/codes/output/predictions_and_actuals.csv')

# Save the DataFrame to CSV in the specified directory
results.to_csv(file_path, index=False)



# Path to the CSV file
file_path = r'/Users/aminahmadisharaf/Downloads/behzad/amin/codes/output/predictions_and_actuals.csv'

# Load the CSV file
data = pd.read_csv(file_path)

# Set global font settings for all plots
plt.rcParams.update({'font.size': 12})

Group = ['a','b','c','d','e','f','g','h','i']
bunch_sample = np.arange(2000, len(data), 2000)

# Create a 3x3 grid of subplots
fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(12, 10))
#log_ks = np.log
# Plot histograms for each group
for i, group in zip(range(9),Group):
    row = i // 3
    col = i % 3
    ax = axs[row, col]
# Ensure no negative or zero values for log transformation
    y_pred_log = (data[f'y_pred_test_{group}'] )  # Add 1 to avoid log10(0)
    y_test_log = (data[f'y_test_{group}'])       # Add 1 to avoid log10(0)

    # Plot histogram for predictions
    ax.hist(y_pred_log, bins=35, alpha=1,density=True, label=r'Predicted K$_{s}$', color='r')
    # Plot histogram for actual values
    ax.hist(y_test_log, bins=35, alpha=1,density=True, label=r'Actual K$_{s}$', color='darkblue')

   
    ax.set_xlabel('')
    ax.set_ylabel('')
    # Set the x-axis labels
    if row == 2 and col == 1:
        ax.set_xlabel('Ln(Ks)',fontsize=16)
    # Set the x-axis tick labels to be bold
 
    #xtick_indices = list(train_sizes)
  
    # Set the number of x-ticks to 5 and show the x-axis numbers corresponding to each y-value
    #ax.set_xticks(train_sizes[xtick_indices])
    #ax.set_xticklabels(train_sizes[xtick_indices])
    
    #ax.set_title(f'{group}', weight='bold')  # position title in lower left corner of subplot
    # Add text box with gray background and position it at left bottom of subplot
    
    props = dict(boxstyle='square', facecolor='white', alpha=0.1)
    ax.text(0.95, 0.93, f'{group}', transform=ax.transAxes, fontweight='bold', bbox=props)
    # Set the y-axis label

    #ax.set_yticks(np.arange(0, 1.05, 0.2))
    if row == 1 and col == 0:
        ax.set_ylabel('Probability',fontsize=16)
    ytick_labels = ax.get_yticklabels()  # Add this line
    plt.setp(ytick_labels)  # Add this lin
    
    
    ax.set_ylim(0, 1)
    ax.set_xlim(-5, 5)
    '''
    
    if row == 0:
        ax.set_ylim(0, 200)  # set y-axis limits
    if row == 1:
            ax.set_ylim(0, 400)  # set y-axis limits
    if row == 2:
            ax.set_ylim(0, 500)  # set y-axis limits
    #ax.tick_params(axis='y', which='both', labelweight='bold')
   '''

    ax.legend(loc='upper left', bbox_to_anchor=(0.0, 1.0))
    # Set the plot title

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the figure
fig.savefig(r'/Users/aminahmadisharaf/Downloads/behzad/amin/codes/output/histograms of predicted and actual values.svg', format='svg')

plt.show()


