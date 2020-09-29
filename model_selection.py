import numpy as np
np.random.seed(143)
import matplotlib.pyplot as plt
from load_data import load_and_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import time


def cross_val_linear_model(train_set):

    X = train_set[:, :-1]  # for all but last column
    y = train_set[:, -1]  # for last column

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    k_best = SelectKBest(f_regression)
    pipeline = Pipeline([('k_best', k_best), ('reg', LinearRegression())])

    params = {
        'reg__fit_intercept': [True, False],
        'k_best__k': [20, 40, 60, 80]
    }

    reg = GridSearchCV(pipeline, params, cv=kf, n_jobs=-1)
    start = time.time()
    reg.fit(X, y)
    end = time.time()
    print('Linear model cross-val time elapsed: ', end - start)

    print(reg.best_params_)
    print("Best cross-val R squared score: %.2f" % reg.best_score_)

    return reg.best_estimator_


def cross_val_xgboost_model(train_set):

    X = train_set[:, :-1]  # for all but last column
    y = train_set[:, -1]  # for last column

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    k_best = SelectKBest(f_regression)
    pipeline = Pipeline([('k_best', k_best), ('reg', XGBRegressor())])

    params = {
        'reg__learning_rate': [0.02, 0.1],
        'reg__subsample': [0.5, 1],
        'reg__max_depth': [6, 16, 20],
        'reg__min_child_weight': [0.5, 1],
        'k_best__k': [20, 40, 60, 80],
    }

    reg = GridSearchCV(pipeline, params, cv=kf, n_jobs=-1)
    start = time.time()
    reg.fit(X, y)
    end = time.time()
    print('XGBoost cross-val time elapsed: ', end - start)

    print(reg.best_params_)
    print("Best cross-val R squared score: %.2f" % reg.best_score_)

    return reg.best_estimator_


def cross_val_ridge_model(train_set):

    X = train_set[:, :-1]  # for all but last column
    y = train_set[:, -1]  # for last column

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    k_best = SelectKBest(f_regression)
    pipeline = Pipeline([('k_best', k_best), ('reg', Ridge())])

    params = {
        'reg__alpha': [0.1, 0.5, 1],
        'reg__fit_intercept': [True, False],
        'k_best__k': [20, 40, 60, 80]
    }

    reg = GridSearchCV(pipeline, params, cv=kf, n_jobs=-1)

    start = time.time()
    reg.fit(X, y)
    end = time.time()
    print('Linear model cross-val time elapsed: ', end - start)
    print(reg.best_params_)
    print("Best cross-val R squared score: %.2f" % reg.best_score_)

    return reg.best_estimator_


def test_model(reg, test_set, plot_graph=False):

    X_test = test_set[:, :-1]  # for all but last column
    y_test = test_set[:, -1]  # for last column

    y_predict = reg.predict(X_test)
    test_r2 = r2_score(y_test, y_predict)
    print("Test R squared score: %.2f" % test_r2)
    if plot_graph:
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_predict)
        ax.plot([0, y_test.max()], [0, y_predict.max()], 'k--', lw=4)
        ax.set_xlabel('Measured')
        ax.set_ylabel('Predicted')
        regressor_name = reg['reg'].__class__.__name__
        fig.savefig(regressor_name)



def train_and_evaluate_linear_model(plot_graph=False):
    data_dict = load_and_split()
    train_set = data_dict.get('train_set').to_numpy()
    test_set = data_dict.get('test_set').to_numpy()
    reg = cross_val_linear_model(train_set)
    test_model(reg, test_set, plot_graph)


def train_and_evaluate_xgboost_model(plot_graph=False):
    data_dict = load_and_split()
    train_set = data_dict.get('train_set').to_numpy()
    test_set = data_dict.get('test_set').to_numpy()
    reg = cross_val_xgboost_model(train_set)
    test_model(reg, test_set, plot_graph)


def train_and_evaluate_ridge_model(plot_graph=False):
    data_dict = load_and_split()
    train_set = data_dict.get('train_set').to_numpy()
    test_set = data_dict.get('test_set').to_numpy()
    reg = cross_val_ridge_model(train_set)
    test_model(reg, test_set, plot_graph)


def validate_linear_model():
    data_dict = load_and_split()
    train_set = data_dict.get('train_set').to_numpy()
    reg = cross_val_linear_model(train_set)


def validate_xgboost_model():
    data_dict = load_and_split()
    train_set = data_dict.get('train_set').to_numpy()
    reg = cross_val_xgboost_model(train_set)


def validate_ridge_model():
    data_dict = load_and_split()
    train_set = data_dict.get('train_set').to_numpy()
    reg = cross_val_ridge_model(train_set)


train_and_evaluate_xgboost_model(plot_graph=True)


