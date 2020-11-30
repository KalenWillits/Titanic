
# __Environment__
import sqlite3 as sql
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import joblib
from tqdm import tqdm
from library import *

cd_data = 'data/'
cd_figures = 'figures/'
cd_docs = 'docs/'
cd_models = 'models/'

db = DataBase(path=cd_data, file_name='titanic.sqlite')

train = db.query("""
    SELECT *
    FROM train;
    """)

test = db.query("""
    SELECT *
    FROM test;
    """)
example = db.query("""
    SELECT *
    FROM example;
    """)

df = process_data(train)

x = df.drop('Survived', axis=1)
y = df.Survived
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# Random Search of model parameters
models = 10000 # Increase this parameter for more randomly generated models.

for model in tqdm(range(models)):
    parameters ={'n_estimators':1,
                'criterion':['gini', 'entropy'][np.random.randint(2)],
                'min_samples_split':np.random.random(),
                'min_samples_leaf':np.random.random()/2,
                'min_weight_fraction_leaf':np.random.random(),
                'max_features':['auto', 'sqrt', 'log2'][np.random.randint(3)],
                'max_leaf_nodes':None,
                'min_impurity_decrease':np.random.random(),
                'min_impurity_split':np.random.random(),
                'bootstrap':np.random.randint(1),
                'oob_score':np.random.randint(1),
                'n_jobs':-1,
                'random_state':42,
                'warm_start':np.random.randint(1),
                'ccp_alpha':0.0,
                'class_weight':[None, 'balanced', 'balanced_subsample'][np.random.randint(3)]}


    rfc = RandomForestClassifier(n_estimators=parameters['n_estimators'],
                criterion=parameters['criterion'],
                min_samples_split=parameters['min_samples_split'],
                min_samples_leaf=parameters['min_samples_leaf'],
                min_weight_fraction_leaf=parameters['min_weight_fraction_leaf'],
                max_features=parameters['max_features'],
                max_leaf_nodes=parameters['max_leaf_nodes'],
                min_impurity_decrease=parameters['min_impurity_decrease'],
                min_impurity_split=parameters['min_impurity_split'],
                bootstrap=parameters['bootstrap'],
                oob_score=parameters['oob_score'],
                n_jobs=parameters['n_jobs'],
                random_state=parameters['random_state'],
                warm_start=parameters['warm_start'],
                ccp_alpha=parameters['ccp_alpha'],
                class_weight=parameters['class_weight'])

    rfc.fit(x_train, y_train)
    y_pred = rfc.predict(x_test)

    # __Model Performance__
    metrics = pd.DataFrame({'accuracy': [accuracy_score(y_test, y_pred)],
    'precision': [precision_score(y_test, y_pred)],
    'recall': [recall_score(y_test, y_pred)],
    'f1': [f1_score(y_test, y_pred)]})

    for key, val in parameters.items():
        metrics[key] = val

# Checks if the model predicted if anyone would survive.
    if y_pred.sum() > 0:
        db.write(metrics, 'rfc-metrics', if_exists='append')


# %% codecell
# Applying the best parameters to the model.

best_parameters = db.query("""
    SELECT DISTINCT * FROM [rfc-metrics]
    WHERE precision > 0
    ORDER BY accuracy DESC;
    """).head(1)

with open(cd_docs+'model_metrics.md', 'w+') as file:
    file.write(best_parameters.to_markdown(index=False))
