

# __Environment__
import sqlite3 as sql
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import joblib
from tqdm import tqdm
from library import *

from logging import captureWarnings
captureWarnings(True)
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

np.random.random(1)

x = df.drop('Survived', axis=1)
y = df.Survived
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# Random Search of model parameters
models = 1000000 # Increase this parameter for more randomly generated models.

for model in tqdm(range(models)):

    parameters = RandomSearchLR()

    lr = LogisticRegression(penalty=parameters.penalty,
                            dual=parameters.dual,
                            tol=parameters.tol,
                            C=parameters.C,
                            fit_intercept=parameters.fit_intercept,
                            intercept_scaling=parameters.intercept_scaling,
                            class_weight=parameters.class_weight,
                            random_state=parameters.random_state,
                            solver=parameters.solver,
                            max_iter=parameters.max_iter,
                            multi_class=parameters.multi_class,
                            verbose=parameters.verbose,
                            warm_start=parameters.warm_start,
                            n_jobs=parameters.n_jobs,
                            l1_ratio=parameters.l1_ratio)


    lr.fit(x_train, y_train)
    y_pred = lr.predict(x_test)

    # __Model Performance__
    metrics = pd.DataFrame({'accuracy': [accuracy_score(y_test, y_pred)],
    'precision': [precision_score(y_test, y_pred)],
    'recall': [recall_score(y_test, y_pred)],
    'f1': [f1_score(y_test, y_pred)]})

    for key, val in parameters.to_dict().items():
        metrics[key] = val

# Checks if the model predicted if anyone would survive.
    if y_pred.sum() > 0:
        db.write(metrics, 'lr_metrics', if_exists='append')


# %% codecell
# Applying the best parameters to the model.

best_parameters = db.query("""
    SELECT DISTINCT * FROM lr_metrics
    WHERE precision > 0
    ORDER BY accuracy DESC;
    """).head(1)

with open(cd_docs+'model_metrics.md', 'w+') as file:
    file.write(best_parameters.to_markdown(index=False))
