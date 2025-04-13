import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
warnings.filterwarnings("ignore")
from scipy.stats import randint, uniform
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Define RMSE as a custom scoring metric

def interpolate_nans(arr):
    """Interpolate NaN values in a 1D NumPy array using linear interpolation."""
    nans, x = np.isnan(arr), lambda z: z.nonzero()[0]
    arr[nans] = np.interp(x(nans), x(~nans), arr[~nans])
    return arr



"""## Read data"""
data = pd.read_csv(r'/Users/aminahmadisharaf/Downloads/behzad/amin/codes/USKSAT_OpenRefined_Cleaned.csv')
accuracy = pd.read_excel(r'/Users/aminahmadisharaf/Downloads/behzad/amin/codes/final_results.xlsx')

R2_Test = accuracy['Test_r2 mean']
RMSE_Test = accuracy['Test_RMSE mean']
R2_Train = accuracy['Train_r2 mean']
RMSE_Train = accuracy['Train RMSE mean']


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

"""## Distributon of ksat columns

"""

r2_aggregated_TestScore =[]
r2_aggregated_TrainScore =[]
bunch_sample = np.arange(2000, 17646, 2000)
bunch_sample= np.append(bunch_sample,len(data))
bunch_sample

fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))
Group = ['a','b','c','d','e','f','g','h','i']
iterator= [0,1,2,3,4,5,6,7,8,9]
for i, bunch, group,  in zip(iterator,bunch_sample,Group):
    
    R2Test = R2_Test[i]
    RMSETest = RMSE_Test[i]
    R2Train = R2_Train[i]
    RMSETrain =RMSE_Train[i]
    
    X = data.loc[:, data.columns != 'Ksat_cmhr']
    y = np.log(data.loc[:, 'Ksat_cmhr'])  # Transform target to ln(Ksat_cmhr + 1) #removed
# Randomly pick bunch of data by 2000 step
    slection = bunch / len(X)
    if slection != 1:
        X, X_test, y, y_test = train_test_split(X,y,test_size=1- slection,random_state=1)
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=1)

    else:
        
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=1)
    
    
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
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time.time() - start), n_iter_search))
    
    best_regressor = random_search.best_estimator_
    random_search.best_params_
    
    """## Get predictions"""
    
    y_pred_test = best_regressor.predict(X_test)
    
    y_pred_train = best_regressor.predict(X_train)
    
    """## Calculate MAE"""
    
    from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_squared_log_error
    
    rmse_pred = mean_absolute_error(y_test, y_pred_test) 
    

    
    
    
    print(f"Mean Squared Error = {mean_squared_error(y_test, y_pred_test)}")
    print("Root Mean Absolute Error = ", np.sqrt(rmse_pred))
    print(f"R-Squared = {r2_score(y_test, y_pred_test)}")
    print("Average log Residual =", np.mean(np.log(np.squeeze(y_pred_test)) - np.log(np.squeeze(y_test))))
    print(f"RMSE = {np.sqrt(np.mean((y_test - y_pred_test)**2))}")
    
    """**Defining & calculating RMSE&R2 for test and train dataset**"""
    
    RMSE_XGBoost_test = "{:.3f}".format(np.sqrt(np.mean((y_test - y_pred_test)**2)))
    RSquared_test = "{:.3f}".format(r2_score(y_test, y_pred_test))
    
    RMSE_XGBoost_train = "{:.3f}".format(np.sqrt(np.mean((y_train - y_pred_train)**2)))
    RSquared_train = "{:.3f}".format(r2_score(y_train, y_pred_train))
    
    RMSE_XGBoost_total = "{:.3f}".format(np.sqrt(np.mean((np.concatenate((y_train, y_test)) - np.concatenate((y_pred_train, y_pred_test)))**2)))
    RSquared_total = "{:.3f}".format(r2_score(np.concatenate((y_train,y_test)), np.concatenate((y_pred_train,y_pred_test))))
    
    any(np.isnan(np.log(np.squeeze(1+ np.concatenate((y_train,y_test))))))
    
    np.sum(np.isnan(np.log(np.squeeze(1 + np.concatenate((y_pred_train,y_pred_test))))))
    
    # learning curve________________________________________________________________
    
    from sklearn.model_selection import learning_curve
    
    Len_X_train = len(X_train)
    train_sizes = list(range(1, int(Len_X_train *(1 - 0.2)) , 1000))
    
    
    """**Defining & calculating RMSE&R2 for test and train dataset**"""
    from sklearn.metrics import make_scorer, mean_squared_log_error

# Define the RMSE scorer
    

#_______________________________________________________________________
# Loop through the subplots and plot the data on a log-log scale

    from scipy.stats import gaussian_kde

    row = i // 3
    col = i % 3
    #font = {'family': 'Times New Roman','size': 10}    
    ax = axs[row, col]
    #plt.rcParams["font.family"] = "Times New Roman"
    FontSize = 12
    
# Calculate the KDE for color
    y_test_values = y_test.values.flatten()
    mask = ~np.isnan(y_test_values) & ~np.isnan(y_pred_test)
    y_test_values = y_test_values[mask]
    y_pred_test_values = y_pred_test[mask]
    
    xy = np.vstack([(y_test_values),(y_pred_test_values)])
    xy = interpolate_nans(xy)

    kde = gaussian_kde(xy)(xy)
    sc = ax.scatter((y_test), (y_pred_test), marker='o', s=2, c=kde)
    line = (-10,8)
    ax.plot(line, line, 'r--', alpha=0.75, zorder=0)
    
    # Adjust the x and y axis limits between min and max of both
    ax.set_xlim([-10, 8])
    ax.set_ylim([-10, 8])
    
    if row == 1 and col == 2:
    # Define a new axes for the colorbar next to the subplot at (1, 2)
    # This assumes that there's enough space in the figure layout to place the colorbar
        fig.subplots_adjust(right=0.85)  # Adjust the right margin to make space for the colorbar
        cbar_ax = fig.add_axes([0.87, 0.3, 0.03, 0.4])  # Modify these values to adjust the colorbar position
        cbar = fig.colorbar(sc, cax=cbar_ax)
        cbar.set_label('Density', fontsize=12)
        

    if row ==2 and col == 1:    
        ax.set_xlabel('Measured log(K$_{s}$ [cm/hr])',weight='bold',fontsize=FontSize)
    xtick_labels = ax.get_xticklabels()  # Add this line
    plt.setp(xtick_labels, weight='bold',fontsize=10)  # Add this lin
    
    # Add text box with gray background and position it at left bottom of subplot
    
    props = dict(boxstyle='square', facecolor='white', alpha=0.1)
    ax.text(0.92, 0.05, f'{group}', transform=ax.transAxes, fontsize=FontSize, fontweight='bold', bbox=props)
    
    props2 = dict(boxstyle='square', facecolor='white', alpha=0.1)
    
    
    
    ax.text(0.05, 0.87, f'R$^2$: {R2Test:.2f}\nRMSLE: {RMSETest:.2f}', 
            transform=ax.transAxes, fontsize=11, fontweight='bold', 
            bbox=props2)
    

    #ax.set_yticks(np.arange(0, 1.2, 0.2),weight='bold')
    if row ==1 and col == 0:
        ax.set_ylabel('Estimated log(K$_{s}$ [cm/hr])',weight='bold',fontsize=16)
    ytick_labels = ax.get_yticklabels()  # Add this line
    plt.setp(ytick_labels, weight='bold',fontsize=10)  # Add this lin





    plt.savefig(f'/Users/aminahmadisharaf/Downloads/behzad/amin/codes/output/loglog density {bunch}.svg',format='svg')
plt.tight_layout(rect=[0, 0, 0.85, 1])  # Leave space on the right side for the colorbar
plt.savefig(r'/Users/aminahmadisharaf/Downloads/behzad/amin/codes/output/loglog density tets.svg',format='svg')

plt.show()

###Traning Set####

fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
Group = ['a','b','c','d','e','f','g','h','i']
#Group = ["ABCDEFGHI"]
iterator= [0,1,2,3,4,5,6,7,8,9]
for i, bunch, group,  in zip(iterator,bunch_sample,Group):
    
    R2Test = R2_Test[i]
    RMSETest = RMSE_Test[i]
    R2Train = R2_Train[i]
    RMSETrain =RMSE_Train[i]
    
    X = data.loc[:, data.columns != 'Ksat_cmhr']
    y = np.log(data.loc[:, 'Ksat_cmhr'])  # Transform target to ln(Ksat_cmhr + 1) #removed
# Randomly pick bunch of data by 2000 step
    slection = bunch / len(X)
    if slection != 1:
        X, X_test, y, y_test = train_test_split(X,y,test_size=1- slection,random_state=1)
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=1)

    else:
        
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=1)
    
    
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
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time.time() - start), n_iter_search))
    
    best_regressor = random_search.best_estimator_
    random_search.best_params_
    
    """## Get predictions"""
    
    y_pred_test = best_regressor.predict(X_test)
    
    y_pred_train = best_regressor.predict(X_train)
    
    """## Calculate MAE"""
    
    from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_squared_log_error
    
    rmse_pred = mean_absolute_error(y_test, y_pred_test) 
    

    
    
    
    print(f"Mean Squared Error = {mean_squared_error(y_test, y_pred_test)}")
    print("Root Mean Absolute Error = ", np.sqrt(rmse_pred))
    print(f"R-Squared = {r2_score(y_test, y_pred_test)}")
    print("Average log Residual =", np.mean(np.log(np.squeeze(y_pred_test)) - np.log(np.squeeze(y_test))))
    print(f"RMSE = {np.sqrt(np.mean((y_test - y_pred_test)**2))}")
    
    """**Defining & calculating RMSE&R2 for test and train dataset**"""
    
    RMSE_XGBoost_test = "{:.3f}".format(np.sqrt(np.mean((y_test - y_pred_test)**2)))
    RSquared_test = "{:.3f}".format(r2_score(y_test, y_pred_test))
    
    RMSE_XGBoost_train = "{:.3f}".format(np.sqrt(np.mean((y_train - y_pred_train)**2)))
    RSquared_train = "{:.3f}".format(r2_score(y_train, y_pred_train))
    
    RMSE_XGBoost_total = "{:.3f}".format(np.sqrt(np.mean((np.concatenate((y_train, y_test)) - np.concatenate((y_pred_train, y_pred_test)))**2)))
    RSquared_total = "{:.3f}".format(r2_score(np.concatenate((y_train,y_test)), np.concatenate((y_pred_train,y_pred_test))))
    
    any(np.isnan(np.log(np.squeeze(1+ np.concatenate((y_train,y_test))))))
    
    np.sum(np.isnan(np.log(np.squeeze(1 + np.concatenate((y_pred_train,y_pred_test))))))
    
    # learning curve________________________________________________________________
    
    
    Len_X_train = len(X_train)
    train_sizes = list(range(1, int(Len_X_train *(1 - 0.2)) , 1000))
    
    


# Define the RMSE scorer
    

#_______________________________________________________________________
# Loop through the subplots and plot the data on a log-log scale

    from scipy.stats import gaussian_kde

    row = i // 3
    col = i % 3
    #font = {'family': 'Times New Roman',
      #      'size': 10}    
    ax = axs[row, col]
   # plt.rcParams["font.family"] = "Times New Roman"
    FontSize = 12
    
# Calculate the KDE for color
    y_train_values = y_train.values.flatten()
    mask = ~np.isnan(y_train_values) & ~np.isnan(y_pred_train)
    y_train_values = y_train_values[mask]
    y_pred_train_values = y_pred_train[mask]
    
    xy = np.vstack([(y_train_values),(y_pred_train_values)])
    xy = interpolate_nans(xy)

    kde = gaussian_kde(xy)(xy)
    sc = ax.scatter((y_train), (y_pred_train), marker='o', s=2, c=kde)
    #ax.scatter(np.log(y_test), np.log(y_test), marker='o', s=2, c='r', label='Actual')    # Add a diagonal line with slope 1 to each subplot
    line = (-10,8)
    ax.plot(line, line, 'r--', alpha=0.75, zorder=0)
    
    # Adjust the x and y axis limits between min and max of both
    ax.set_xlim([-10, 8])
    ax.set_ylim([-10, 8])
    
    if row == 1 and col == 2:
    # Define a new axes for the colorbar next to the subplot at (1, 2)
    # This assumes that there's enough space in the figure layout to place the colorbar
        fig.subplots_adjust(right=0.85)  # Adjust the right margin to make space for the colorbar
        cbar_ax = fig.add_axes([0.87, 0.3, 0.03, 0.4])  # Modify these values to adjust the colorbar position
        cbar = fig.colorbar(sc, cax=cbar_ax)
        cbar.set_label('Density', fontsize=12)
        

    if row ==2 and col == 1:    
        ax.set_xlabel('Measured log(K$_{s}$ [cm/hr])',weight='bold',fontsize=FontSize)
    xtick_labels = ax.get_xticklabels()  # Add this line
    plt.setp(xtick_labels, weight='bold',fontsize=10)  # Add this lin
    
    # Add text box with gray background and position it at left bottom of subplot
    
    props = dict(boxstyle='square', facecolor='white', alpha=0.1)
    ax.text(0.92, 0.05, f'{group}', transform=ax.transAxes, fontsize=FontSize, fontweight='bold', bbox=props)
    
    #props2 = dict(boxstyle='square', facecolor='white', alpha=0.1)
    #        ax.text(0.05, 0.87, f'R$^2$: {R2Test:.2f}\nRMSLE: {RMSETest:.2f}', 
    #        transform=ax.transAxes, fontsize=FontSize, fontweight='bold', 
    #        bbox=props2)
    
    props2 = dict(boxstyle='square', facecolor='white', alpha=0.1)
    ax.text(0.05, 0.87, f'R$^2$: {R2Train:.2f}\nRMSLE: {RMSETrain:.2f}', transform=ax.transAxes, fontsize=11, fontweight='bold', bbox=props2)
    # Set the y-axis label

    #ax.set_yticks(np.arange(0, 1.2, 0.2),weight='bold')
    if row ==1 and col == 0:
        ax.set_ylabel('Estimated log(K$_{s}$ [cm/hr])',weight='bold',fontsize=FontSize)
    ytick_labels = ax.get_yticklabels()  # Add this line
    plt.setp(ytick_labels, weight='bold',fontsize=10)  # Add this lin






    
    plt.savefig(f'/Users/aminahmadisharaf/Downloads/behzad/amin/codes/output/training loglog density {bunch}.svg',format='svg')
plt.tight_layout(rect=[0, 0, 0.85, 1])  # Leave space on the right side for the colorbar
plt.savefig(r'/Users/aminahmadisharaf/Downloads/behzad/amin/codes/output/loglog density train.svg',format='svg')
