import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
warnings.filterwarnings("ignore")
from sklearn.model_selection import RandomizedSearchCV
import time
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import randint, uniform

data = pd.read_csv(r'/Users/aminahmadisharaf/Downloads/behzad/amin/codes/USKSAT_OpenRefined_Cleaned.csv')

def isnotequal(x, y):
    if abs(x - y) < 10**-2:
        return False
    return True

invalidindecies = np.array(list(map(isnotequal, data.iloc[:, 11:16].sum(axis=1), data.iloc[:, 16])))
data = data[~invalidindecies]
columns_to_check = ['Ksat_cmhr', 'Db', 'OC', 'Clay', 'Silt', 'Sand', 'VCOS', 'COS', 'MS', 'FS', 'VFS', 'Depth.cm_Top', 'Depth.cm_Bottom', 'Dia.cm', 'Height.cm']
data = data.dropna(subset=columns_to_check)
numeric_columns = ['Silt', 'Clay', 'Sand', 'Depth.cm_Top', 'Depth.cm_Bottom', 'Height.cm', 'Dia.cm', 'Db', 'VCOS', 'COS', 'MS', 'FS', 'VFS', 'Ksat_cmhr']
for col in numeric_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')
data = data.drop(columns=['Ref', 'Site', 'Soil', 'Sand', 'Field', 'Method', 'Depth.cm_Bottom', 'Dia.cm', 'Height.cm'])
is_NaN = data.isnull()
row_has_NaN = is_NaN.any(axis=1)
rows_with_NaN = data[row_has_NaN]
print(rows_with_NaN)

"""## Select Features and Target"""

train_RMSE_mean_list = []
train_RMSE_std_list = []
test_RMSE_mean_list = []
test_RMSE_std_list = []
train_r2_mean_list = []
train_r2_std_list = []
test_r2_mean_list = []
test_r2_std_list = []

Hyper_parameters = []
bunch_sample = np.arange(2000, len(data), 2000)
bunch_sample = np.append(bunch_sample, len(data))
bunch_iteration = []

Group = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
iterator = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
for i, bunch, group in zip(iterator, bunch_sample, Group):
    X = data.loc[:, data.columns != 'Ksat_cmhr']
    y = np.log(data.loc[:, 'Ksat_cmhr'])  # Transform target to ln(Ksat_cmhr + 1) #removed
    if bunch / len(X) != 1:
        X, X_test, y, y_test = train_test_split(X, y, test_size=1 - bunch / len(X), random_state=1)
    sc = StandardScaler()
    X.values[:, :] = sc.fit_transform(X.values[:, :])
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
    random_search = RandomizedSearchCV(reg, param_distributions=params, n_iter=n_iter_search, cv=5, scoring='neg_mean_squared_error')
    start = time.time()
    train_RMSE = []
    test_RMSE = []
    train_r2 = []
    test_r2 = []
    param_df = pd.DataFrame(columns=['iteration', 'n_estimators', 'min_child_weight', 'gamma', 'learning_rate',
                                     'subsample', 'colsample_bytree', 'max_depth', 'objective', 'booster'])
    for t in range(50):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=t+1)
        random_search.fit(X_train, np.ravel(y_train))
        best_params = random_search.best_params_
        best_params['iteration'] = t+1
        param_df = pd.concat([param_df, pd.DataFrame([best_params])], ignore_index=True)
        best_regressor = random_search.best_estimator_
        y_pred_test = best_regressor.predict(X_test)
        y_pred_train = best_regressor.predict(X_train)
        RMSE_test = mean_squared_error(y_test, y_pred_test, squared=False)
        RMSE_train = mean_squared_error(y_train, y_pred_train, squared=False)
        r2_test = r2_score(y_test, y_pred_test)
        r2_train = r2_score(y_train, y_pred_train)
        train_RMSE.append(RMSE_train)
        test_RMSE.append(RMSE_test)
        train_r2.append(r2_train)
        test_r2.append(r2_test)
    train_RMSE_mean = np.mean(train_RMSE)
    train_RMSE_std = np.std(train_RMSE)
    test_RMSE_mean = np.mean(test_RMSE)
    test_RMSE_std = np.std(test_RMSE)
    train_r2_mean = np.mean(train_r2)
    train_r2_std = np.std(train_r2)
    test_r2_mean = np.mean(test_r2)
    test_r2_std = np.std(test_r2)
    Hyper_parameters.append(param_df)
    train_RMSE_mean_list.append(train_RMSE_mean)
    train_RMSE_std_list.append(train_RMSE_std)
    test_RMSE_mean_list.append(test_RMSE_mean)
    test_RMSE_std_list.append(test_RMSE_std)
    train_r2_mean_list.append(train_r2_mean)
    train_r2_std_list.append(train_r2_std)
    test_r2_mean_list.append(test_r2_mean)
    test_r2_std_list.append(test_r2_std)
    print(i)
    bunch_iteration.append(bunch)
final_results = {'samplesize': bunch_iteration, 'Train RMSE mean': train_RMSE_mean_list, 'Train RMSE std': train_RMSE_std_list,
                 'Test_RMSE mean': test_RMSE_mean_list, 'Test RMSE std': test_RMSE_std_list,
                 'Train_r2 mean': train_r2_mean_list, 'Train_r2 std': train_r2_std_list,
                 'Test_r2 mean': test_r2_mean_list, 'Test_r2 std': test_r2_std_list,
                 'Hyper parameters': Hyper_parameters}
df = pd.DataFrame.from_dict(final_results)


# Save final results to an Excel file
final_results_path = r'/Users/aminahmadisharaf/Downloads/behzad/amin/codes/final_results.xlsx'
df.to_excel(final_results_path, index=False)

# Save hyperparameters to another Excel file, each set in a different sheet
hyper_parameters_path = r'/Users/aminahmadisharaf/Downloads/behzad/amin/codes/hyper_parameters.xlsx'
with pd.ExcelWriter(hyper_parameters_path) as writer:
    for idx, params_df in enumerate(Hyper_parameters):
        sheet_name = Group[idx]  # Group labels are 'a' to 'i'
        params_df.to_excel(writer, sheet_name=sheet_name, index=False)
