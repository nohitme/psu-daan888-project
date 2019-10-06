# Neural network
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import neural_network
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# override display settings to show all columns
pd.set_option('display.max_columns', 500)
# don't use scientific notation
pd.set_option('display.float_format', lambda x: '%.6f' % x)

# do oni_2.csv for now
data = pd.read_csv('oni_2.csv')

# drop index column
data = data.drop(data.columns[0], axis=1)

# Linear regression
# move Anom to the end and remove weather_class for easier operations later
cname_anom = 'Anom'
cname_weather_class = 'weather_class'

cols = list(data.columns.values)
cols.pop(cols.index(cname_anom))
cols.pop(cols.index(cname_weather_class))
data = data[cols + [cname_anom]]

# generate dummies
data_dummies = pd.get_dummies(data)
# print(data_dummies.dtypes)

# Split data set
train, test = train_test_split(data_dummies, test_size=0.3, random_state=87)

# Rescale data to [0, 1]
scaler = MinMaxScaler()
scaler.fit(train)
train_scaled = scaler.transform(train)
test_scaled = scaler.transform(test)

anom_index = train.columns.get_loc(cname_anom)
y_train_scaled = train_scaled[:, anom_index]
x_train_scaled = np.delete(train_scaled, anom_index, 1)
y_test_scaled = test_scaled[:, anom_index]
x_test_scaled = np.delete(test_scaled, anom_index, 1)


# custom report function for metrics
def metrics_report(model, x_train, y_train, x_test, y_test):
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    r2_train = metrics.r2_score(y_train, y_train_pred)
    mse_train = metrics.mean_squared_error(y_train, y_train_pred)
    print(f"R2 Score (train): {r2_train:.4f}")
    print(f"MSE (train) : {mse_train:.4f}")

    r2_test = metrics.r2_score(y_test, y_test_pred)
    mse_test = metrics.mean_squared_error(y_test, y_test_pred)
    print(f"R2 Score (test): {r2_test:.4f}")
    print(f"MSE (test) : {mse_test:.4f}")

    print()
    print(f"R2 Ratio (test / train): {(r2_test / r2_train):.4f}")
    print(f"MSE Ratio (test / train): {(mse_test / mse_train):.4f}")
    print()
    explained = explained_variance_score(y_test, y_test_pred);
    print(f"Explained variance: {explained:.4f}")
    print()


# custom cross validation reports
def get_cross_val_metrics(model, x, y):
    scoring = ['explained_variance',
               'neg_mean_squared_error',
               'r2']
    scores = cross_validate(model, x, y, scoring=scoring,
                            cv=10, return_train_score=True)
    return pd.DataFrame(scores).mean()


nnet = neural_network.MLPRegressor(30, max_iter=20000, random_state=77)
mean_score = get_cross_val_metrics(nnet, x_train_scaled, y_train_scaled)
print(f"cross val score: {mean_score}")
print()


# Optimization
def get_optimizer(model, tuned_parameters, x, y):
    optimizer = GridSearchCV(model,
                             tuned_parameters,
                             scoring="neg_mean_squared_error",
                             cv=10,
                             return_train_score=False,
                             n_jobs=-1,
                             verbose=10)
    optimizer.fit(x, y)
    score_lists = ['param_' + k for k in tuned_parameters.keys()]
    score_lists.extend(['mean_test_score', 'std_test_score', 'rank_test_score'])
    results = pd.DataFrame(optimizer.cv_results_)[score_lists]
    return optimizer, results


nn_parameters = {
    'hidden_layer_sizes': list(range(10, 90, 5)),
    'activation': ['logistic', 'tanh', 'relu'],
    'solver': ['lbfgs', 'sgd', 'adam'],
}

nno = neural_network.MLPRegressor(max_iter=20000, random_state=77)
nn_optimizer, nn_results = get_optimizer(nno, nn_parameters, x_train_scaled, y_train_scaled)
print(nn_results.sort_values(by='rank_test_score').head(20))
print(nn_optimizer.best_params_)
print(nn_optimizer.best_estimator_)

metrics_report(nn_optimizer.best_estimator_, x_train_scaled, y_train_scaled, x_test_scaled, y_test_scaled)
