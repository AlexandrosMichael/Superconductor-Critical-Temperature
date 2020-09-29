import numpy as np

np.random.seed(143)
import matplotlib.pyplot as plt
from load_data import load_and_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import r2_score
import time


# method which carries out the cross validation and grid search process for the random forest model using the training set
# returns the best estimator resulting from the grid search
def cross_val_rf_model(train_set):
    X = train_set[:, :-1]  # for all but last column
    y = train_set[:, -1]  # for last column

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    k_best = SelectKBest(f_regression)
    pipeline = Pipeline([('k_best', k_best), ('reg', RandomForestRegressor())])

    # these are the grid search parameters
    params = {
        'reg__n_estimators': [10, 100, 200],
        'reg__max_features': ['auto', 'sqrt'],
        'reg__min_samples_split': [2, 5],
        'reg__min_samples_leaf': [1, 2],
        'reg__bootstrap': [False, True],
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


# method used to carry out the random forest model selection process
def validate_rf_model():
    data_dict = load_and_split()
    train_set = data_dict.get('train_set').to_numpy()
    reg = cross_val_rf_model(train_set)


# this method evaluates the performance of an estimator passed as a parameter
# it prints out the test scores and it will also plot a scatter plot of actual vs predicted values if plot_graph=True
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


# method used to carry out the random forest model selection process and also the evaluation process on the test set
def train_and_evaluate_rf_model(plot_graph=False):
    data_dict = load_and_split()
    train_set = data_dict.get('train_set').to_numpy()
    test_set = data_dict.get('test_set').to_numpy()
    reg = cross_val_rf_model(train_set)
    test_model(reg, test_set, plot_graph)


train_and_evaluate_rf_model(plot_graph=True)
