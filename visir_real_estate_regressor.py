from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import tree
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml                        # using openml to import dataset 
#from pandas_profiling import ProfileReport                       
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer                   # to transform column of different types
from sklearn.model_selection import train_test_split            
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV                # to find best hyper parameters
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
# import metrics
from sklearn.metrics import plot_confusion_matrix
#from sklearn.metrics import regression_report

# To display multiple output from a single cell.
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


from sklearn.metrics import r2_score

json_file = 'dataset/fastV2.json'
df = pd.read_json(json_file)

del df["image"]
del df["images_nr"]
del df["openhouse"]
del df["sale_or_rent"]
del df["longitude"]
del df["latitude"]
del df["id"]

df["town"] = df["zip"].apply(lambda x: x["town"])
df["zip"] = df["zip"].apply(lambda x: x["zip"])

df["fastmat"] = 0
df["garage"] = False
df["lyft"] = False
df["view"] = False
df["bilageymsla"] = False
df["fokhelt"] = False
df["hofud"] = False
df["nr"] = 1
#df["fastmat"] = "0"
#df = df.assign(garage=lambda x: str(x["town"]))
df['size'] = df['size'].str.replace(',', '.').astype(float)

#df.iloc[:,:].str.replace(',', '.').astype(float)

for index, row in df.iterrows():
    if "fasteignamat</span><span class=\"data\">" in row["extra_info"].lower():
        
        ind = str(row["extra_info"]).find("""Fasteignamat""") + 38
        
        s = str(row["extra_info"])[ind: ind+11]
        s = s.replace(".", "")
        s = s.replace("k", "")
        s = s.replace("r", "")
        try:
            a = float(s)
            df.loc[index, "fastmat"] = a
        except:
            pass
        

    if "lyft" in row["description"].lower() or "lyft" in row["extra_info"].lower():
        df.loc[index, "lyft"] = True
        df.loc[index, "nr"] += 1
    if "úts\u00fdni" in row["description"].lower():
        df.loc[index, "view"] = True
        df.loc[index, "nr"] += 1
    if "fokhelt" in row["description"].lower():
        df.loc[index, "fokhelt"] = True
        df.loc[index, "nr"] -= 1
    if int(row["zip"]) in [101,102,103,104,105,106,107,108,109,110,111,112,113,116,162,200,201,202,203,210,212,225,220,221,222,270,271,276,170]:
        df.loc[index, "hofud"] = True
        df.loc[index, "nr"] += 1
    if "b\u00edlsk\u00far" in row["description"].lower() or "B\u00edlsk\u00far" in row["extra_info"].lower() :
        df.loc[index, "garage"] = True
        df.loc[index, "nr"] += 2
    elif "b\u00edlageym" in row["description"].lower():
        df.loc[index, "bilageymsla"] = True
        df.loc[index, "nr"] += 1

    

#del df["rooms"] 
del df["lyft"] 
#del df["view"] 
del df["nr"] 
#del df["garage"] 
#del df["bilageymsla"] 
#del df["fokhelt"]
del df["description"] 
del df["extra_info"] 
del df["street_name"] 
del df["street_number"]
#del df["hofud"]
#df = df[df['street_number'] == df['street_number']]
#del df["category"]
del df["legit_realestate_agent"]



# remove NaN
#df = df[df['longitude'] == df['longitude']]
df = df[df['make_year'] == df['make_year']]
#df = df[df['make_year'] > 2000]
df = df[df['rooms'] == df['rooms']]


#df = df[df['bathrooms'] == df['bathrooms']]
#df = df[df['bedrooms'] == df['bedrooms']]
df = df[df['category'] != "Atvinnuh\u00fasn\u00e6\u00f0i"]
df = df[df['category'] != "L\u00f3\u00f0"]
df = df[df['category'] != "Hesthús"]
#df = df[df['category'] == "Fjölbýlishús"]


del df["fastmat"] 
df = df[df['price'] != 0] 



del df["bedrooms"]
del df["bathrooms"]
#del df["category"]

#df.size = df.size.astype(float)

df.info(verbose=True)


plt.figure(figsize=(10, 6))
sns.distplot(df['price'],bins=30)

#plt.show()

print(df.head())


print(df.shape)
print(df.columns)
print(df.dtypes)

plt.figure(figsize=(14, 8))
corr_matrix = df.corr().round(2)
sns.heatmap(data=corr_matrix,cmap='coolwarm',annot=True)

y = df.pop("price")

X = df
#X = df.to_dict("records")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# Reporting on df


print(df)
print(df.columns)

print(df.shape)



print(df.shape)
print(df.columns)
print(df.dtypes)

plt.figure(figsize=(14, 8))
corr_matrix = df.corr().round(2)
sns.heatmap(data=corr_matrix,cmap='coolwarm',annot=True)

# Numerical features in the data
numerical_features = ["size","make_year","rooms"]

# categorical features in the data
categorical_features = ["zip",'town','category']
boolean = ['hofud',"garage","bilageymsla","fokhelt","view"]

# min max scaler ?

# build pipeline

numerical_transformer = Pipeline(steps=[
   # ('imputer', SimpleImputer()),
    ('scaler', 'passthrough')])

categorical_transformer = Pipeline(steps=[
   # ('imputer', SimpleImputer()),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

data_transformer = ColumnTransformer(
    transformers=[
        ('numerical', numerical_transformer, numerical_features),
        ('categorical', categorical_transformer, categorical_features)])

preprocessor = Pipeline(steps=[('data_transformer', data_transformer)])
#preprocessor = Pipeline(steps=[('data_transformer', data_transformer), ('reduce_dim',TruncatedSVD())])

#classifier = Pipeline(steps=[('preprocessor', preprocessor),
#                      ('classifier', LogisticRegression(random_state=0, max_iter=10000))])
#classifier = Pipeline(steps=[('classifier', DecisionTreeRegressor())])
classifier = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestRegressor())])


# hypertuning random forest tree
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
#n_estimators=100, *, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None)[source]¶


param_grid = {
#    'preprocessor__data_transformer__numerical__imputer__strategy': ['mean', 'median'],
#    'preprocessor__data_transformer__categorical__imputer__strategy': ['constant','most_frequent'],
    'preprocessor__data_transformer__numerical__scaler': [StandardScaler(), RobustScaler(), \
                                                          MinMaxScaler()],
    #'classifier__C': [0.1, 1.0, 10, 100],
    #'preprocessor__reduce_dim__n_components': [1,2, 3,5],
    'classifier__n_estimators': n_estimators,
    'classifier__max_features': max_features,
    'classifier__max_depth': max_depth,
    'classifier__max_features': max_features,
    'classifier__min_samples_split': min_samples_split,
    'classifier__min_samples_leaf': min_samples_leaf,
    'classifier__bootstrap': bootstrap,

} 
#X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


# grid_search = GridSearchCV(classifier, param_grid=param_grid,cv = 5,) # setja inn custom scoring 
rf = RandomForestRegressor()
grid_search = RandomizedSearchCV(estimator = classifier, param_distributions = param_grid, n_iter = 50, cv = 10, verbose=1, n_jobs = -1)


grid_search.fit(X_train, y_train)

from sklearn import set_config

# set config to diagram for visualizing the pipelines
set_config(display='diagram')

# lets see our best estimator

# saving pipeling as html format

y_pred = grid_search.predict(X_test)
print("train accuracy: %f" % grid_search.score(X_train, y_train))
print("test accuracy: %f" % grid_search.score(X_test, y_test))



real = y_test.tolist()
# make a prediction
pre = grid_search.predict(X_test)
a = []
for i in range(30):
    print("predict: ",pre[i], ", Real: ", real[i])
    #a.append(abs(float(str(pre[i]))-float(real[i])))
    
from scipy.stats import sem
import numpy as np
import sklearn.metrics as metrics
print(np.mean(a)/1000000)
import math
def evaluate(model, test_features, test_labels,train_features,train_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape

    mae = metrics.mean_absolute_error(test_labels, predictions)
    mse = metrics.mean_squared_error(test_labels, predictions)

    print('Model Performance')

    print("TEST SET \n==============")
    print('R2 score {}'.format(model.score(test_features, test_labels)))
    print('Accuracy =           {:0.2f}%.'.format(accuracy))
    print('Mean absoloute error {}'.format(mae))
    print("RMSE:                {}".format(math.sqrt(mse)))
    print('Average Error:       {}'.format(np.mean(errors)))


    print("TRAINING SET \n==============")


    # TRAINING ACCURACY


    train_predictions = model.predict(train_features)
    train_errors = abs(train_predictions - train_labels)
    train_mape = 100 * np.mean(train_errors / train_labels)
    train_accuracy = 100 - train_mape

    train_mae = metrics.mean_absolute_error(train_labels, train_predictions)
    train_mse = metrics.mean_squared_error(train_labels, train_predictions)

    print('R2 score {}'.format(model.score(train_features, train_labels)))
    print('Accuracy =           {:0.2f}%.'.format(train_accuracy))
    print('Mean absoloute error {}'.format(train_mae))
    print("RMSE:                {}".format(math.sqrt(train_mse)))
    print('Average Error:       {}'.format(np.mean(train_errors)))



    print("\n Features:         ")

    print("Best estimator", model.best_estimator_)
    print("Best params", model.best_params_, "\n")


    return accuracy

evaluate(grid_search, X_test, y_test, X_train, y_train)
