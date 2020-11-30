# %% markdown
# Kaggle's Titanic ML challenge
## Table Of Contents:
# - [Connecting To The Database](#Connecting-To-The-Database)
# - [Exploratory Data Analysis](#Exploratory-Data-Analysis)
# - [Observations](#Observations)
# - [Model Performance](#Model-Performance)
# - [Final Model](#Final-Model)
# - [Model Observations](#Model-Observations)

# %% markdown
## Connecting To The Database

# %% codecell
# __Environment__
import sqlite3 as sql
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import joblib
from tqdm import tqdm
import sys
import numba
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

# %% markdown
# Observations
#  There are some standout metrics from our report. Even though there were less
# women on the Titanic, women still had a better chance of survival than men.
# Ranking at only 35% of the population on board, 68% of women survived in our
#  training data.
#  There are is also a remarkably high survival rate among passengers who had
# Higher priced tickets, specifically ones who where in the 50%-75% range of pricing.
def model_test():
    # %% codecell
    # __Data Modeling__
    x = df.drop('Survived', axis=1)
    y = df.Survived
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

    # Random Search of model parameters
    models = 100 # Increase this parameter for more randomly generated models.

    for model in tqdm(range(models)):
        c = np.random.random()*np.random.randint(2)
        parameters = {'C':c if c > 0 else 0.1,
         'kernel':'poly',
         'degree':np.random.randint(10),
         'gamma':['scale', 'auto'][np.random.randint(2)],
         'coef0':np.random.random()*np.random.randint(2),
         'shrinking':np.random.randint(2),
         'probability':np.random.randint(2),
         'tol':np.random.random()/100,
         'cache_size':200,
         'class_weight':None,
         'verbose':False,
         'max_iter':-1,
         'decision_function_shape':'ovr',
         'break_ties':np.random.randint(2),
         'random_state':42}

        svc = SVC(C=parameters['C'],
            kernel=parameters['kernel'],
            degree=parameters['degree'],
            gamma=parameters['gamma'],
            coef0=parameters['coef0'],
            shrinking=parameters['shrinking'],
            probability=parameters['probability'],
            tol=parameters['tol'],
            cache_size=parameters['cache_size'],
            class_weight=parameters['class_weight'],
            verbose=parameters['verbose'],
            max_iter=parameters['max_iter'],
            decision_function_shape=parameters['decision_function_shape'],
            break_ties=parameters['break_ties'],
            random_state=parameters['random_state'])

        svc.fit(x_train, y_train)
        y_pred = svc.predict(x_test)

        # __Model Performance__
        metrics = pd.DataFrame({'accuracy': [accuracy_score(y_test, y_pred)],
        'precision': [precision_score(y_test, y_pred)],
        'recall': [recall_score(y_test, y_pred)],
        'f1': [f1_score(y_test, y_pred)]})

        for key, val in parameters.items():
            metrics[key] = val

    # Checks if the model predicted if anyone would survive.
        if y_pred.sum() > 0:
            db.write(metrics, 'svc-metrics', if_exists='append')


    # %% codecell
    # Applying the best parameters to the model.

    best_parameters = db.query("""
        SELECT DISTINCT * FROM [svc-metrics]
        WHERE precision > 0
        ORDER BY accuracy DESC;
        """).head(1)

    with open(cd_docs+'model_metrics.md', 'w+') as file:
        file.write(best_parameters.to_markdown(index=False))

    # %% markdown
    # ## Model Performance
    # |   accuracy |   precision |   recall |       f1 |        C | kernel   |   degree | gamma   |    coef0 |   shrinking |   probability |        tol |   cache_size | class_weight   |   verbose |   max_iter | decision_function_shape   |   break_ties |   random_state |
    # |-----------:|------------:|---------:|---------:|---------:|:---------|---------:|:--------|---------:|------------:|--------------:|-----------:|-------------:|:---------------|----------:|-----------:|:--------------------------|-------------:|---------------:|
    # |    0.80339 |    0.798077 | 0.691667 | 0.741071 | 0.154449 | poly     |        2 | auto    | 0.797978 |           0 |             1 | 0.00974807 |          200 |                |         0 |         -1 | ovr                       |            0 |             42 |
    # This has provided a good start to answering our problem. We know that if we continue our random search we are bound to get better results. However due to the computational limitaions we have opted to go with this model.
    # Perhaps we could attempt this notebook again using an Nvidia GPU with cuda cores.

    # %% markdown
    # ## Final Model
    # Automated query to best parameters according to accuracy and applying them to the production model.

    # %% codecell
    # __Model Production__
    svc = SVC(C=best_parameters['C'].values[0],
        kernel=best_parameters['kernel'].values[0],
        degree=best_parameters['degree'].values[0],
        gamma=best_parameters['gamma'].values[0],
        coef0=best_parameters['coef0'].values[0],
        shrinking=best_parameters['shrinking'].values[0],
        probability=best_parameters['probability'].values[0],
        tol=best_parameters['tol'].values[0],
        cache_size=best_parameters['cache_size'].values[0],
        class_weight=best_parameters['class_weight'].values[0],
        verbose=best_parameters['verbose'].values[0],
        max_iter=best_parameters['max_iter'].values[0],
        decision_function_shape=best_parameters['decision_function_shape'].values[0],
        break_ties=best_parameters['break_ties'].values[0],
        random_state=best_parameters['random_state'].values[0])

    # %% codecell
    # Pulling train and test data again.
    train = db.query("""
        SELECT *
        FROM train;
        """)

    test = db.query("""
        SELECT *
        FROM test;
        """)

    # Preparing the data for prediction
    p_test = process_data(test)
    p_train = process_data(train)
    x = p_train.drop('Survived', axis=1)
    y = p_train.Survived

    # Traing and saving the model
    svc.fit(x, y)
    joblib.dump(svc, cd_models+'svc.pkl')

    # There is a single null value in the test data/Fare column!!
    # - Replacing it with the median.
    p_test.Fare.fillna(p_test.Fare.median(), inplace=True)
    y_pred = svc.predict(p_test)
    # Need to add the passengerID back in as per the submission requirements.
    pred = pd.DataFrame({'PassengerId':p_test.PassengerId, 'Survived':y_pred})
    db.write(pred, 'prediction')
    pred.to_csv(cd_data+'titanic-prediction.csv', index=False)


    # %% markdown
    # ## Model Observations.
    # The production model on the test data produced a score on the [Kaggle Leaderboards](https://www.kaggle.com/c/titanic/leaderboard) of 0.75119.
    # The next step would be to allow a GPU to run 1000 or more iterations of this model to gain optimal parameters.
    #
    # There is evidence that supports what type of passenger survived to the titanic with this data. The first stand out metric is sex. A female passenger has a ~68% chance of survival when compared to a male passenger at ~32%. Now the distribution of male vs female passengers is not equal at ~65% male vs ~35% female. This means a model that does nothing but assume all female passengers survive according to our training data will be correct ~68% of the time. We know from the example submission that this is not the case. The example submission achieves a score of ~77%. This means our model is over-fit to the training data and we should re-evaluate our testing and training set. So we can assume that 77% of the passengers left on Kaggle's testing data is female.
    #
    # According to [titanic facts](https://titanicfacts.net/titanic-lifeboats/) there were 20 lifeboats on board which could only carry 33% of total passengers to safety. According to this training data ~38% of the ship survived. That plus the fact that [the lifeboats were not used at full capacity](https://www.historyonthenet.com/the-titanic-lifeboats) means our training data distribution is skewed towards the optimistic side.
    #
    # This does not bode well for our over-fit SVC. Further parameter tuning and more model exploration is required.
@cuda.jit
model_test()
