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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

data = pd.read_csv(r'/Users/aminahmadisharaf/Downloads/behzad/amin/codes/USKSAT_OpenRefined_Cleaned.csv')

def isnotequal (x, y):
    if abs(x - y) < 10**-2:
        return False
    return True

invalidindecies = np.array(list(map(isnotequal, data.iloc[: , 11:16].sum(axis=1), data.iloc[: , 16])))
data = data[~invalidindecies]

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

data = data.drop(columns=['Ref', 'Site', 'Soil', 'Sand', 'Field', 'Method', 'Depth.cm_Bottom', 'Dia.cm', 'Height.cm'])

is_NaN = data.isnull()
row_has_NaN = is_NaN.any(axis=1)
rows_with_NaN = data[row_has_NaN]
print(rows_with_NaN)

param_df = pd.DataFrame(columns=['iteration', 'n_estimators', 'min_child_weight', 'gamma', 'learning_rate', 
                                 'subsample', 'colsample_bytree', 'max_depth', 'objective', 'booster'])

r2_aggregated_TestScore = []
r2_aggregated_TrainScore = []
bunch_sample = np.arange(2000, len(data), 2000)
bunch_sample= np.append(bunch_sample,len(data))
fig, axs = plt.subplots(3, 3, figsize=(15, 15))
Group = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
iterator = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
for i, bunch, group in zip(iterator, bunch_sample, Group):
    X = data.loc[:, data.columns != 'Ksat_cmhr']
    y = np.log(data.loc[:, 'Ksat_cmhr'])
    slection = bunch / len(X)
    if slection != 1:
        X, X_test, y, y_test = train_test_split(X, y, test_size=1 - slection, random_state=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        print(len(X_train))
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        print(len(X_train))

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train.values[:, :] = sc.fit_transform(X_train.values[:, :])
    X_test.values[:, :] = sc.transform(X_test.values[:, :])
    
    from xgboost import XGBRegressor
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import RandomizedSearchCV
    import time
    
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
    n_iter_search = 100
    random_search = RandomizedSearchCV(reg,
                                       param_distributions=params,
                                       n_iter=n_iter_search,
                                       cv=5,
                                       scoring='neg_mean_squared_error')
    
    start = time.time()
    random_search.fit(X_train, y_train)
    
    best_params = random_search.best_params_
    param_df = pd.concat([param_df, pd.DataFrame([best_params])], ignore_index=True)
    best_regressor = random_search.best_estimator_
    
    y_pred_test  = best_regressor.predict(X_test)
    y_pred_train = best_regressor.predict(X_train)
    
    rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)  # Calculating RMSE
    rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)  # Calculating RMSE
    
    print(f"Root Mean Squared Error for Test Set: {rmse_test}")
    print(f"Root Mean Squared Error for Train Set: {rmse_train}")
    print(f"R-Squared for Test Set: {r2_score(y_test, y_pred_test)}")
    
    from sklearn.model_selection import learning_curve
    
    Len_X_train = len(X_train)
    train_sizes = np.linspace(5, Len_X_train, 10, dtype=int)
    
    train_sizes, train_scores, test_scores = learning_curve(
        best_regressor,  # the model
        X_train,  # training data
        y_train,  # training targets
        cv=5,  # cross-validation
        train_sizes=np.linspace(0.1, 1.0, 10),  # training sizes
        scoring='r2',  # scoring metric
        n_jobs=-1  # number of CPUs
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    r2_aggregated_TestScore.append(test_scores)
    r2_aggregated_TrainScore.append(train_scores)
    
    row = i // 3
    col = i % 3
    ax = axs[row, col]
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({'font.size': 12})
    ax.plot(train_sizes, train_mean, label='Train', color='maroon', marker='o')
    ax.plot(train_sizes, test_mean, label='Cross-validation', color='darkblue', linestyle='--', marker='s')
    
    if row == 2 and col == 1:
        ax.set_xlabel('Number of samples', weight='bold')
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontweight('bold')
    xtick_indices = list(train_sizes)
    ax.set_xticks(train_sizes[xtick_indices])
    ax.set_xticklabels(train_sizes[xtick_indices])
    
    props = dict(boxstyle='square', facecolor='white', alpha=0.1)
    ax.text(0.05, 0.05, f'{group}', transform=ax.transAxes, fontweight='bold', bbox=props)

    if row == 1 and col == 0:
        ax.set_ylabel('R$^2$', weight='bold')

    ytick_labels = ax.get_yticklabels()  # Get y-tick labels
    plt.setp(ytick_labels, weight='bold')  # Set y-tick labels to bold
    ax.set_ylim(0, 1.05)  # Set y-axis limits

    ax.legend(loc='lower right', bbox_to_anchor=(1.0, 0.0), prop={'weight':'bold'})

# Set overall labels for the figures
# Adjust subplot settings and apply uniform axis labels
# Adjust subplot settings and apply uniform axis labels and styles
# Adjust subplot settings and apply uniform axis labels and styles
for ax in axs.flat:
    # Set x and y axis tick labels to bold
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')

    # Set labels with bold font weight for axis
    ax.set_xlabel('Number of samples', weight='bold')
    ax.set_ylabel('R$^2$', weight='bold')

    # Make sure labels are only shown on the outer edges of the grid to avoid label redundancy on internal plots
    ax.label_outer()

# Complete all plot configurations and then save the figure
fig.savefig(r'/Users/aminahmadisharaf/Downloads/behzad/amin/codes/output/learning_curve_final_colorful_modified_test.svg', format='svg')
plt.show()




