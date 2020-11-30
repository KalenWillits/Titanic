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
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import joblib
from tqdm import tqdm
from library import *
rm = np.random

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

# # %% codecell
# plt.title('Heatmap Of Titanic Data Correlations')
# sns.heatmap(df.corr())
# plt.savefig(cd_figures+'heatmap', transparent=True)
#
# # %% markdown
# # ### Heatmap Observation
# # The heatmap shows a strong correlation with sex and surviving.
# # However, there are also weak correlations with Fare, Cabin, and location as well.
# # We know that cabin and fare are related. This is because the higher level cabins
# # cost more.
# # ### Questions:
# # - What is the rate of women that survived vs the rate of men that survived?
# # - Does having children increase the rate of survival?
# # - Does the boarding location effect survival rate?
#
# # %% codecell
# #__DB Queries__
#
# df_survived = db.query("""
#     SELECT *
#     FROM train_clean
#     WHERE Survived = True;
#     """)
#
#
# df_families = db.query("""
#     SELECT *
#     FROM train_clean
#     WHERE Parch > 0
#     AND SibSp > 0;
#     """)
#
# df_families_survived = db.query("""
#     SELECT *
#     FROM train_clean
#     WHERE Parch > 0
#     AND SibSp > 0
#     AND Survived = True;
#     """)
#
# df_solo = db.query("""
#     SELECT *
#     FROM train_clean
#     WHERE Parch = 0
#     AND SibSp = 0;
#     """)
#
# df_solo_survived = db.query("""
#     SELECT *
#     FROM train_clean
#     WHERE Parch = 0
#     AND SibSp = 0
#     AND Survived = True;
#     """)
#
# df_cherbourg = db.query("""
#     SELECT *
#     FROM train_clean
#     WHERE Cherbourg = True;
#     """)
#
# df_cherbourg_survived = db.query("""
#     SELECT *
#     FROM train_clean
#     WHERE Cherbourg = True
#     AND Survived = True;
#     """)
#
# df_queenstown = db.query("""
#     SELECT *
#     FROM train_clean
#     WHERE Queenstown = True;
#     """)
#
# df_queenstown = db.query("""
#     SELECT *
#     FROM train_clean
#     WHERE Queenstown = True;
#     """)
#
# df_queenstown_survived = db.query("""
#     SELECT *
#     FROM train_clean
#     WHERE Queenstown = True
#     AND Survived = True;
#     """)
#
# df_southampton = db.query("""
#     SELECT *
#     FROM train_clean
#     WHERE Southampton = True;
#     """)
#
# df_southampton_survived = db.query("""
#     SELECT *
#     FROM train_clean
#     WHERE Southampton = True
#     AND Survived = True;
#     """)
#
# q1 = df.Fare.std()
# q2 = df.Fare.std()*2
# q3 = df.Fare.std()*3
# q4 = df.Fare.std()*4
#
# df_fareQ1 = db.query("""
#     SELECT *
#     FROM train_clean
#     WHERE FareBinned <= {0};
#     """.format(q1))
#
# df_fareQ2 = db.query("""
#     SELECT *
#     FROM train_clean
#     WHERE FareBinned > {0}
#     AND FareBinned <= {1};
#     """.format(q1, q2))
#
# df_fareQ3 = db.query("""
#     SELECT *
#     FROM train_clean
#     WHERE FareBinned > {0}
#     AND FareBinned <= {1};
#     """.format(q2, q3))
#
# df_fareQ4 = db.query("""
#     SELECT *
#     FROM train_clean
#     WHERE FareBinned > {0};
#     """.format(q3))
#
# df_fareQ1_survived = db.query("""
#     SELECT *
#     FROM train_clean
#     WHERE FareBinned < {0}
#     AND Survived = True;
#     """.format(q1))
#
# df_fareQ2_survived = db.query("""
#     SELECT *
#     FROM train_clean
#     WHERE FareBinned > {0}
#     AND FareBinned <= {1}
#     AND Survived = True;
#     """.format(q2, q3))
#
# df_fareQ3_survived = db.query("""
#     SELECT *
#     FROM train_clean
#     WHERE FareBinned > {0}
#     AND FareBinned <= {1}
#     AND Survived = True;
#     """.format(q3, q4))
#
# df_fareQ4_survived = db.query("""
#     SELECT *
#     FROM train_clean
#     WHERE FareBinned > {0}
#     AND Survived = True;
#     """.format(q3))
#
# # %% codecell
# # __Calculations__
# ratio_survived = df_survived.shape[0] / df.shape[0]
#
# num_women = sum(df['IsFemale'])
# num_men = sum(df['IsMale'])
# num_total = df.shape[0]
#
# ratio_women = num_women / num_total
# ratio_men = num_men / num_total
#
# num_women_survived = sum(df_survived['IsFemale'])
# num_men_survived = sum(df_survived['IsMale'])
# num_survived = num_women_survived + num_men_survived
# ratio_women_survived = num_women_survived / num_survived
# ratio_men_survived = num_men_survived / num_survived
#
# families = df_families.shape[0]
# families_survived = df_families_survived.shape[0]
# solo = df_solo.shape[0]
# solo_survived = df_solo_survived.shape[0]
#
# ratio_families = families / num_total
# ratio_families_survived = families_survived / families
# ratio_solo = solo / num_total
# ratio_solo_survived = solo_survived / solo
#
# cherbourg = df_cherbourg.shape[0]
# cherbourg_survived = df_cherbourg_survived.shape[0]
# queenstown = df_queenstown.shape[0]
# queenstown_survived = df_queenstown_survived.shape[0]
# southampton = df_southampton.shape[0]
# southampton_survived = df_southampton_survived.shape[0]
#
# ratio_cherbourg = cherbourg / num_total
# ratio_cherbourg_survived = cherbourg_survived / cherbourg
# ratio_queenstown = queenstown / num_total
# ratio_queenstown_survived = queenstown_survived / queenstown
# ratio_southampton = southampton / num_total
# ratio_southampton_survived = southampton_survived / southampton
#
# ratio_lowF = df_fareQ1.shape[0] / num_total
# ratio_medF = df_fareQ2.shape[0] / num_total
# ratio_highF = df_fareQ3.shape[0] / num_total
# ratio_veryhighF = df_fareQ4.shape[0] / num_total
#
# ratio_lowF_survived = df_fareQ1_survived.shape[0] / df_fareQ1.shape[0]
# ratio_medF_survived = df_fareQ2_survived.shape[0] / df_fareQ2.shape[0]
# ratio_highF_survived = df_fareQ3_survived.shape[0] / df_fareQ3.shape[0]
# ratio_veryhighF_survived = df_fareQ4_survived.shape[0] / df_fareQ4.shape[0]
#
# # %% codecell
# # __Data Visualization__
# shift = 3
# outlier_gate_index = -10
# title = "Distrobution Of Fare Prices"
# plt.title(title)
# plt.hist(df.Fare, color = 'black', bins=10)
# plt.vlines(np.arange(shift, q4, q1+shift), 0, max([df_fareQ1.shape[0]]), color='cyan')
# plt.savefig(cd_figures+title.lower().replace(' ', '-'), transparent=True)
#
# # __Reporting__
# report = ('# __Titanic Exploratory Data Analysis__' +
# '\nRatio of survivors: {0}'.format(round(ratio_survived,2)) +
# '\nRatio of female passengers: {0}'.format(round(ratio_women,2)) +
# '\nRatio of female survivors: {0}'.format(round(ratio_women_survived, 2)) +
# '\nRatio of male passengers: {0}'.format(round(ratio_men, 2)) +
# '\nRatio of male survivors: {0}'.format(round(ratio_men_survived, 2)) +
# '\nRatio of family members: {0}'.format(round(ratio_families, 2)) +
# '\nRatio of family members that survived: {0}'.format(round(ratio_families_survived, 2)) +
# '\nRatio of solo passengers: {0}'.format(round(ratio_solo, 2)) +
# '\nRatio of solo passengers that survived: {0}'.format(round(ratio_solo_survived, 2)) +
# '\nRatio of passengers embarked from Cherbourg: {0}'.format(round(ratio_cherbourg, 2)) +
# '\nRatio of passengers embarked from Queenstown: {0}'.format(round(ratio_queenstown, 2)) +
# '\nRatio of passengers embarked from Southampton: {0}'.format(round(ratio_southampton, 2)) +
# '\nRatio of embarkees from Cherbourg that survived: {0}'.format(round(ratio_cherbourg_survived, 2)) +
# '\nRatio of embarkees from Queenstown that survived: {0}'.format(round(ratio_queenstown_survived, 2)) +
# '\nRatio of embarkees from Southampton that survived: {0}'.format(round(ratio_southampton_survived, 2)) +
# '\nRatio of low fares: {0}'.format(round(ratio_lowF, 2)) +
# '\nRatio of medium fares: {0}'.format(round(ratio_medF, 2)) +
# '\nRatio of high fares: {0}'.format(round(ratio_highF, 2)) +
# '\nRatio of very high fares: {0}'.format(round(ratio_veryhighF, 2)) +
# '\nRatio of low fares that survived: {0}'.format(round(ratio_lowF_survived, 2)) +
# '\nRatio of medium fares that survived: {0}'.format(round(ratio_medF_survived, 2)) +
# '\nRatio of high fares that survived: {0}'.format(round(ratio_highF_survived, 2)) +
# '\nRatio of very high fares that survived: {0}'.format(round(ratio_veryhighF_survived, 2)))
#
# with open(cd_docs+'EDA.md', 'w+') as doc:
#     doc.write(report)

# %% markdown
# Observations
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
models = 1 # Increase this parameter for more randomly generated models.

for model in tqdm(range(models)):
    parameters ={'n_estimators':1,
                'criterion':['gini', 'entropy'][np.random.randint(2)],
                'min_samples_split':np.random.randint(5)*np.random.random(),
                'min_samples_leaf':np.random.randint(3)*np.random.random(),
                'min_weight_fraction_leaf':np.random.random(),
                'max_features':['auto', 'sqrt', 'log2'][np.random.random.randint(3)],
                'max_leaf_nodes':None,
                'min_impurity_decrease':np.random.random(),
                'min_impurity_split':np.random.random(),
                'bootstrap':np.random.randint(1),
                'oob_score':np.random.randint(1),
                'n_jobs':-1,
                'random_state':42,
                'wanp.random_start':np.random.randint(1),
                'ccp_alpha':0.0,
                'class_weight':[None, 'balanced', 'balanced_subsample'][np.random.randint(3)]}
}

    rfc = RandomForestClassifier(n_estimators=parameters['n_estimators'],
                criterion=parameters['criterion'],
                min_samples_split=parameters['min_samples_split'],
                min_samples=parameters['min_samples_leaf'],
                min_weight_fraction_leaf=parameters['min_weight_fraction'],
                max_features=parameters['max_features'],
                max_leaf_nodes=parameters['max_leaf_nodes'],
                min_impurity_decrease=parameters['min_impurity_decrease'],
                min_impurity_split=parameters['min_impurity_split'],
                bootstrap=parameters['bootstrap'],
                oob_score=parameters['oob_score'],
                n_jobs=parameters['n_jobs'],
                random_state=parameters['random_state'],
                warm_start=parameters['warm_start'],
                ccp_aplha=parameters['ccp_alpha'],
                class_weight=parameters['class_weight'])

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
# svc = SVC(C=best_parameters['C'].values[0],
#     kernel=best_parameters['kernel'].values[0],
#     degree=best_parameters['degree'].values[0],
#     gamma=best_parameters['gamma'].values[0],
#     coef0=best_parameters['coef0'].values[0],
#     shrinking=best_parameters['shrinking'].values[0],
#     probability=best_parameters['probability'].values[0],
#     tol=best_parameters['tol'].values[0],
#     cache_size=best_parameters['cache_size'].values[0],
#     class_weight=best_parameters['class_weight'].values[0],
#     verbose=best_parameters['verbose'].values[0],
#     max_iter=best_parameters['max_iter'].values[0],
#     decision_function_shape=best_parameters['decision_function_shape'].values[0],
#     break_ties=best_parameters['break_ties'].values[0],
#     random_state=best_parameters['random_state'].values[0])
#
# # %% codecell
# # Pulling train and test data again.
# train = db.query("""
#     SELECT *
#     FROM train;
#     """)
#
# test = db.query("""
#     SELECT *
#     FROM test;
#     """)
#
# # Preparing the data for prediction
# p_test = process_data(test)
# p_train = process_data(train)
# x = p_train.drop('Survived', axis=1)
# y = p_train.Survived
#
# # Traing and saving the model
# svc.fit(x, y)
# joblib.dump(svc, cd_models+'svc.pkl')
#
# # There is a single null value in the test data/Fare column!!
# # - Replacing it with the median.
# p_test.Fare.fillna(p_test.Fare.median(), inplace=True)
# y_pred = svc.predict(p_test)
# # Need to add the passengerID back in as per the submission requirements.
# pred = pd.DataFrame({'PassengerId':p_test.PassengerId, 'Survived':y_pred})
# db.write(pred, 'prediction')
# pred.to_csv(cd_data+'titanic-prediction.csv', index=False)
#

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
