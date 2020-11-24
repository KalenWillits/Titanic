# %% markdown
# Kaggle's Titanic ML challenge
## Table Of Contents:
# - [Connecting To The Database](#Connecting-To-The-Database)
# - [Exploratory Data Analysis](#Exploratory-Data-Analysis)

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

# %% codecell
# __Cleaning The Train Data__
df = process_data(train)
db.write(df, 'train_clean')
# %% markdown
## Exploratory Data Analysis

# %% codecell
plt.title('Heatmap Of Titanic Data Correlations')
sns.heatmap(df.corr())
plt.savefig(cd_figures+'heatmap', transparent=True)

# %% markdown
# ### Heatmap Observation
# The heatmap shows a strong correlation with sex and surviving.
# However, there are also weak correlations with Fare, Cabin, and location as well.
# We know that cabin and fare are related. This is because the higher level cabins
# cost more.
# ### Questions:
# - What is the rate of women that survived vs the rate of men that survived?
# - Does having children increase the rate of survival?
# - Does the boarding location effect survival rate?

# %% codecell
#__DB Queries__

df_survived = db.query("""
    SELECT *
    FROM train_clean
    WHERE Survived = True;
    """)


df_families = db.query("""
    SELECT *
    FROM train_clean
    WHERE Parch > 0
    AND SibSp > 0;
    """)

df_families_survived = db.query("""
    SELECT *
    FROM train_clean
    WHERE Parch > 0
    AND SibSp > 0
    AND Survived = True;
    """)

df_solo = db.query("""
    SELECT *
    FROM train_clean
    WHERE Parch = 0
    AND SibSp = 0;
    """)

df_solo_survived = db.query("""
    SELECT *
    FROM train_clean
    WHERE Parch = 0
    AND SibSp = 0
    AND Survived = True;
    """)

df_cherbourg = db.query("""
    SELECT *
    FROM train_clean
    WHERE Cherbourg = True;
    """)

df_cherbourg_survived = db.query("""
    SELECT *
    FROM train_clean
    WHERE Cherbourg = True
    AND Survived = True;
    """)

df_queenstown = db.query("""
    SELECT *
    FROM train_clean
    WHERE Queenstown = True;
    """)

df_queenstown = db.query("""
    SELECT *
    FROM train_clean
    WHERE Queenstown = True;
    """)

df_queenstown_survived = db.query("""
    SELECT *
    FROM train_clean
    WHERE Queenstown = True
    AND Survived = True;
    """)

df_southampton = db.query("""
    SELECT *
    FROM train_clean
    WHERE Southampton = True;
    """)

df_southampton_survived = db.query("""
    SELECT *
    FROM train_clean
    WHERE Southampton = True
    AND Survived = True;
    """)

q1 = df.Fare.std()
q2 = df.Fare.std()*2
q3 = df.Fare.std()*3
q4 = df.Fare.std()*4

df_fareQ1 = db.query("""
    SELECT *
    FROM train_clean
    WHERE FareBinned <= {0};
    """.format(q1))

df_fareQ2 = db.query("""
    SELECT *
    FROM train_clean
    WHERE FareBinned > {0}
    AND FareBinned <= {1};
    """.format(q1, q2))

df_fareQ3 = db.query("""
    SELECT *
    FROM train_clean
    WHERE FareBinned > {0}
    AND FareBinned <= {1};
    """.format(q2, q3))

df_fareQ4 = db.query("""
    SELECT *
    FROM train_clean
    WHERE FareBinned > {0};
    """.format(q3))

df_fareQ1_survived = db.query("""
    SELECT *
    FROM train_clean
    WHERE FareBinned < {0}
    AND Survived = True;
    """.format(q1))

df_fareQ2_survived = db.query("""
    SELECT *
    FROM train_clean
    WHERE FareBinned > {0}
    AND FareBinned <= {1}
    AND Survived = True;
    """.format(q2, q3))

df_fareQ3_survived = db.query("""
    SELECT *
    FROM train_clean
    WHERE FareBinned > {0}
    AND FareBinned <= {1}
    AND Survived = True;
    """.format(q3, q4))

df_fareQ4_survived = db.query("""
    SELECT *
    FROM train_clean
    WHERE FareBinned > {0}
    AND Survived = True;
    """.format(q3))

# %% codecell
# __Calculations__
num_women = sum(df['IsFemale'])
num_men = sum(df['IsMale'])
num_total = df.shape[0]

ratio_women = num_women / num_total
ratio_men = num_men / num_total

num_women_survived = sum(df_survived['IsFemale'])
num_men_survived = sum(df_survived['IsMale'])
num_survived = num_women_survived + num_men_survived
ratio_women_survived = num_women_survived / num_survived
ratio_men_survived = num_men_survived / num_survived

families = df_families.shape[0]
families_survived = df_families_survived.shape[0]
solo = df_solo.shape[0]
solo_survived = df_solo_survived.shape[0]

ratio_families = families / num_total
ratio_families_survived = families_survived / families
ratio_solo = solo / num_total
ratio_solo_survived = solo_survived / solo

cherbourg = df_cherbourg.shape[0]
cherbourg_survived = df_cherbourg_survived.shape[0]
queenstown = df_queenstown.shape[0]
queenstown_survived = df_queenstown_survived.shape[0]
southampton = df_southampton.shape[0]
southampton_survived = df_southampton_survived.shape[0]

ratio_cherbourg = cherbourg / num_total
ratio_cherbourg_survived = cherbourg_survived / cherbourg
ratio_queenstown = queenstown / num_total
ratio_queenstown_survived = queenstown_survived / queenstown
ratio_southampton = southampton / num_total
ratio_southampton_survived = southampton_survived / southampton

ratio_lowF = df_fareQ1.shape[0] / num_total
ratio_medF = df_fareQ2.shape[0] / num_total
ratio_highF = df_fareQ3.shape[0] / num_total
ratio_veryhighF = df_fareQ4.shape[0] / num_total

ratio_lowF_survived = df_fareQ1_survived.shape[0] / df_fareQ1.shape[0]
ratio_medF_survived = df_fareQ2_survived.shape[0] / df_fareQ2.shape[0]
ratio_highF_survived = df_fareQ3_survived.shape[0] / df_fareQ3.shape[0]
ratio_veryhighF_survived = df_fareQ4_survived.shape[0] / df_fareQ4.shape[0]

# %% codecell
# __Data Visualization__
shift = 3
outlier_gate_index = -10
title = "Distrobution Of Fare Prices"
plt.title(title)
plt.hist(df.Fare, color = 'black', bins=10)
plt.vlines(np.arange(shift, q4, q1+shift), 0, max([df_fareQ1.shape[0]]), color='cyan')
plt.savefig(cd_figures+title.lower().replace(' ', '-'), transparent=True)

# __Reporting__
report = ('# __Titanic Exploratory Data Analysis__' +
'\nRatio of female passengers: {0}'.format(round(ratio_women,2)) +
'\nRatio of female survivors: {0}'.format(round(ratio_women_survived, 2)) +
'\nRatio of male passengers: {0}'.format(round(ratio_men, 2)) +
'\nRatio of male survivors: {0}'.format(round(ratio_men_survived, 2)) +
'\nRatio of family members: {0}'.format(round(ratio_families, 2)) +
'\nRatio of family members that survived: {0}'.format(round(ratio_families_survived, 2)) +
'\nRatio of solo passengers: {0}'.format(round(ratio_solo, 2)) +
'\nRatio of solo passengers that survived: {0}'.format(round(ratio_solo_survived, 2)) +
'\nRatio of passengers embarked from Cherbourg: {0}'.format(round(ratio_cherbourg, 2)) +
'\nRatio of passengers embarked from Queenstown: {0}'.format(round(ratio_queenstown, 2)) +
'\nRatio of passengers embarked from Southampton: {0}'.format(round(ratio_southampton, 2)) +
'\nRatio of embarkees from Cherbourg that survived: {0}'.format(round(ratio_cherbourg_survived, 2)) +
'\nRatio of embarkees from Queenstown that survived: {0}'.format(round(ratio_queenstown_survived, 2)) +
'\nRatio of embarkees from Southampton that survived: {0}'.format(round(ratio_southampton_survived, 2)) +
'\nRatio of low fares: {0}'.format(round(ratio_lowF, 2)) +
'\nRatio of medium fares: {0}'.format(round(ratio_medF, 2)) +
'\nRatio of high fares: {0}'.format(round(ratio_highF, 2)) +
'\nRatio of very high fares: {0}'.format(round(ratio_veryhighF, 2)) +
'\nRatio of low fares that survived: {0}'.format(round(ratio_lowF_survived, 2)) +
'\nRatio of medium fares that survived: {0}'.format(round(ratio_medF_survived, 2)) +
'\nRatio of high fares that survived: {0}'.format(round(ratio_highF_survived, 2)) +
'\nRatio of very high fares that survived: {0}'.format(round(ratio_veryhighF_survived, 2)))

with open(cd_docs+'EDA.md', 'w+') as doc:
    doc.write(report)

# %% markdown
#  There are some standout metrics from our report. Even though there were less
# women on the Titanic, women still had a better chance of survival than men.
# Ranking at only 35% of the population on board, 68% of women survived in our
#  training data.
#  There are is also a remarkably high survival rate among passengers who had
# Higher priced tickets, specifically ones who where in the 50%-75% range of pricing.

# %% codecell
# __Data Modeling__
x = df.drop('Survived', axis=1)
y = df.Survived
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# Random Search of model parameters
models = 1

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

p_test = process_data(test)
p_train = process_data(train)
x = p_train.drop('Survived', axis=1)
y = p_train.Survived
svc.fit(x, y)
joblib.dump(svc, cd_models+'svc.pkl')
# There is a single null value in the test data/Fare column!!
# - Replacing it with the median.
p_test.isnull().sum()
p_test.median()
p_test.Fare.fillna(p_test.Fare.median(), inplace=True)
y_pred = svc.predict(p_test)
# Need to add the passengerID back in here.
pred = pd.DataFrame({'PassengerId':p_test.PassengerId, 'Survived':y_pred})
db.write(df_y_pred, 'prediction')
pred.to_csv(cd_data+'titanic-prediction.csv', index=False)

# Kaggle submission accuracy: 0.75119
