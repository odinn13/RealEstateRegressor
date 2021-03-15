import math
import pickle
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics as metrics
from IPython.core.interactiveshell import InteractiveShell
from pandas_profiling import ProfileReport
from sklearn import set_config, tree
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, r2_score
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     train_test_split)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (MinMaxScaler, OneHotEncoder, RobustScaler,
                                   StandardScaler)
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import estimator_html_repr

InteractiveShell.ast_node_interactivity = "all"


def evaluate(model, test_features, test_labels, train_features, train_labels):
    result = {}

    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape

    mae = metrics.mean_absolute_error(test_labels, predictions)
    mse = metrics.mean_squared_error(test_labels, predictions)

    result["Test"] = {"R2": model.score(
        test_features, test_labels), "accuracy": accuracy, "mae": mae, "rmse": math.sqrt(mse), "ave": np.mean(errors), "best_estim": model.best_estimator_, "best_param": model.best_params_}

    print('Model Performance')

    print("TEST SET \n==============")
    print('R2 score {}'.format(result["Test"]["R2"]))
    print('Accuracy =           {:0.2f}%.'.format(result["Test"]["accuracy"]))
    print('Mean absoloute error {}'.format(result["Test"]["mae"]))
    print("RMSE:                {}".format(result["Test"]["rmse"]))
    print('Average Error:       {}'.format(result["Test"]["ave"]))

    # TRAINING ACCURACY

    train_predictions = model.predict(train_features)
    train_errors = abs(train_predictions - train_labels)
    train_mape = 100 * np.mean(train_errors / train_labels)
    train_accuracy = 100 - train_mape

    train_mae = metrics.mean_absolute_error(train_labels, train_predictions)
    train_mse = metrics.mean_squared_error(train_labels, train_predictions)
    print("Train SET \n==============")
    result["Train"] = {"R2": model.score(
        train_features, train_labels), "accuracy": train_accuracy, "mae": train_mae, "rmse": math.sqrt(train_mse), "ave": np.mean(train_errors), "best_estim": model.best_estimator_, "best_param": model.best_params_, "result": model}

    print('R2 score {}'.format(result["Train"]["R2"]))
    print('Accuracy =           {:0.2f}%.'.format(result["Train"]["accuracy"]))
    print('Mean absoloute error {}'.format(result["Train"]["mae"]))
    print("RMSE:                {}".format(result["Train"]["rmse"]))
    print('Average Error:       {}'.format(result["Train"]["ave"]))

    print("\n Features:         ")

    print("Best estimator", result["Test"]["best_estim"])
    print("Best params", result["Test"]["best_param"], "\n")

    return result


def import_data_and_create_df(all_data):
    df_fjolb_hbsv = pd.read_csv("dataset/fjolb_hbsv.csv", sep=";")
    df_fjolb_landsb = pd.read_csv("dataset/fjolb_landsb.csv", sep=";")

    df = pd.concat([df_fjolb_hbsv, df_fjolb_landsb], ignore_index=True)
    df['utgdag'] = pd.to_datetime(df['utgdag'])

    if not all_data:
        df = df[(df['utgdag'] < pd.Timestamp(
            '2020-01-01')) & (df['utgdag'] > pd.Timestamp('2019-01-01'))]

    return df


def report_on_dataframe(df):
    # Reporting on df

    print(df.dtypes)

    print(df)
    print(df.columns)

    print(df.shape)

    profile_report = ProfileReport(df)
    name_of_pr = 'profiling_report.html'
    profile_report.to_file(name_of_pr)
    print("Profile report in file:", name_of_pr)
    plt.figure(figsize=(14, 8))
    corr_matrix = df.corr().round(2)
    sns.heatmap(data=corr_matrix, cmap='coolwarm', annot=True)

    plt.show()


def generate_pipeline(numerical_features, categorical_features, model):
    numerical_transformer = Pipeline(steps=[
        ('scaler', 'passthrough')])

    # categorical_transformer = Pipeline(steps=[
    #   ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    data_transformer = ColumnTransformer(
        transformers=[
            ('numerical', numerical_transformer, numerical_features),
            #    ('categorical', categorical_transformer, categorical_features)
        ])

    preprocessor = Pipeline(steps=[('data_transformer', data_transformer)])

    regressor = Pipeline(steps=[('preprocessor', preprocessor),
                                ('regressor', model)])

    return regressor


def powspace(start, stop, power, num):
    start = np.power(start, 1/float(power))
    stop = np.power(stop, 1/float(power))
    return np.power(np.linspace(start, stop, num=num), power)


def hyperparameters_for_RFR():

    # hypertuning random forest tree
    # Number of trees in random forest
    n_estimators = [int(x) for x in powspace(
        start=10, stop=3000, power=2, num=20)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt', 'log2']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in powspace(10, 150, 2, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [int(x) for x in powspace(2, 20, 2, num=10)]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [int(x) for x in powspace(1, 20, 2, num=10)]
    # Method of selecting samples for training each tree
    # bootstrap = [True, False]

    param_grid = {
        'preprocessor__data_transformer__numerical__scaler': [StandardScaler(), RobustScaler(),
                                                              MinMaxScaler()],
        'regressor__n_estimators': n_estimators,
        'regressor__max_features': max_features,
        'regressor__max_depth': max_depth,
        'regressor__min_samples_split': min_samples_split,
        'regressor__min_samples_leaf': min_samples_leaf,
        # 'regressor__bootstrap': bootstrap,
        'regressor__n_jobs': [-1]

    }
    return param_grid


def hyperparameters_for_KNR():
    k = [1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 25, 40, 60, 100]
    # to do: laga
    metric = ["minkowski",
              # "chebyshev",
              "manhattan",
              "euclidean"]

    param_grid = {
        'preprocessor__data_transformer__numerical__scaler': [StandardScaler(), RobustScaler(),
                                                              MinMaxScaler()],
        'regressor__n_neighbors': k,
        'regressor__metric': metric,
        'regressor__n_jobs': [-1]
    }
    return param_grid


def tune_parameters_and_get_best_estimator(regressor, param_grid, scorer, cv, iter, X_train, y_train, random_search=True):

    if random_search:
        grid_search = RandomizedSearchCV(estimator=regressor, param_distributions=param_grid,
                                         cv=cv, n_jobs=-1, verbose=1, n_iter=iter, scoring=scorer, return_train_score=True)
    else:
        grid_search = GridSearchCV(estimator=regressor, param_grid=param_grid,
                                   cv=cv, n_jobs=-1, verbose=1, scoring=scorer, return_train_score=True)
    grid_search.fit(X_train, y_train)

    set_config(display='diagram')
    # saving pipeling as html format
    with open(f"price_data_pipeline_estimator{str(datetime.now()).replace(' ','').replace('.','').replace(':','_')}.html", 'w') as f:
        f.write(estimator_html_repr(grid_search.best_estimator_))
    return grid_search


def train_model_and_tune_parameters(regressor, param_grid, X_train, y_train, X_test, y_test, random_search=True):
    score_list = {"R2 SCORER": metrics.make_scorer(metrics.r2_score)
                  # ,
                  #          "MAE SCORER": metrics.make_scorer(metrics.mean_absolute_error, greater_is_better=False),
                  #        "MSE SCORER": metrics.make_scorer(metrics.mean_squared_error, greater_is_better=False),
                  #        "medAE SCORER": metrics.make_scorer(metrics.median_absolute_error, greater_is_better=False)
                  }
    for key, scorer in score_list.items():
        print("=====")
        print(key)

    grid_search = tune_parameters_and_get_best_estimator(
        regressor=regressor, param_grid=param_grid, scorer=scorer, iter=400, cv=10, X_train=X_train, y_train=y_train, random_search=random_search)

    y_pred = grid_search.predict(X_test)

    result = evaluate(model=grid_search, test_features=X_test,
                      test_labels=y_test, train_features=X_train, train_labels=y_train)

    return result


def main():
    args = sys.argv

    all_data = False
    if "all" in args:
        all_data = True

    df = import_data_and_create_df(all_data)

    if "report" in args:
        report_on_dataframe(df)
        exit()

    # Splitting the dataframe into labels and features
    if "predict_square_price" in args:
        y = df.nuvirdi / df.ibm2
        del df["nuvirdi"]
    else:
        y = df.pop("nuvirdi")

    X = df

    # Feature extraction on date
    df['year'] = df['utgdag'].dt.year
    df['month'] = df['utgdag'].dt.month

    # Remove unescesary features and features that have coefficient of 1
    del df["kaupverd"]
    del df["utgdag"]

    # all numerical features of the dataset
    numerical_features = ["ibm2", "bath_fixtures", 'ib2m2', 'ib3m2', 'rism2', 'bilskurm2',
                          'geymm2', 'svalm2', 'haednr', 'fjibmhl', 'fjbilsk',
                          'lodpflm', 'innig_factor', "hverfi", 'ist120', 'age_studull', 'gata', 'top_floor',
                          'two_storey', 'storeys_3', 'lyftuhus', 'nybygging', 'fjarmalastofnunselur', 'year', 'month']
    categorical_features = []

    # Splitting data into train and test datasets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)

    # RandomForestRegressor
    RFR_regressor = generate_pipeline(
        numerical_features=numerical_features, categorical_features=categorical_features, model=RandomForestRegressor())
    RFR_param_grid = hyperparameters_for_RFR()

    # Knearest
    KNR_regressor = generate_pipeline(
        numerical_features=numerical_features, categorical_features=categorical_features, model=KNeighborsRegressor())
    KNR_param_grid = hyperparameters_for_KNR()

    models = {
        "RFR": {"regressor": RFR_regressor, "param_grid": RFR_param_grid, "random_search": True},
        "KNR": {"regressor": KNR_regressor, "param_grid": KNR_param_grid, "random_search": False}
    }

    result = {}
    for model, param in models.items():
        result[model] = train_model_and_tune_parameters(
            regressor=param["regressor"], param_grid=param["param_grid"], X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, random_search=param["random_search"])
    # saves results and model to file
    file_to_write = open(
        f"output{str(datetime.now()).replace(' ','').replace('.','').replace(':','_')}.pickle", "wb")

    pickle.dump(result, file_to_write)


main()
