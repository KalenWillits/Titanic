import sqlite3 as sql
import pandas as pd

cd_data = 'data/'
db = 'titanic.sqlite'

train = pd.read_csv(cd_data+'train.csv')
test = pd.read_csv(cd_data+'test.csv')

with sql.connect(db) as con:
    train.to_sql(train, con, if_exists='replace')
    test.to_sql(train, con, if_exists='replace')
