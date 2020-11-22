import numpy as np
import pandas as pd
import sqlite3 as sql

class DataBase:
    def __init__(self, file_name='db.sqlite', path=''):
        """
        Class to interact with sqlite database more effiecently during data
        analysis.
        """
        self.file_name = file_name
        self.path = path


    def query(self, query_string):
        """
        Queries the SQLite database and returns it as a Pandas DataFrame.
        """
        conn = sql.connect(self.path+self.file_name)
        df = pd.read_sql_query(query_string, conn)
        conn.close()
        return df

    def write(self, df, table_name, if_exists='replace'):
        """
        Writes a Pandas DataFrame to the SQLite database.
        table_name: String that defines the table's name.
        if_exists: Default replaces the table with the given DataFrame. See
        Pandas documentation for the DataFrame.to_sql method.
        """

        conn = sql.connect(self.path+self.file_name)
        df.to_sql(table_name, conn, index=False, if_exists=if_exists)
        conn.commit()
        conn.close()


def strip_to_first(data):
    """
    Strips a pandas series full of strings down to the first character only.
    """
    new_data = []
    for row in data:
        row = row.strip(' ')
        if row == '':
            new_data.append('Z')
        else:
            new_data.append(row[0])
    new_data_ps = pd.Series(new_data)
    return new_data_ps

def process_data(data):
    # Numeric values
    data.PassengerId = pd.to_numeric(data.PassengerId)
    data.Survived = pd.to_numeric(data.Survived)
    data.Pclass = pd.to_numeric(data.Pclass)
    data.Age = pd.to_numeric(data.Age)
    data.SibSp = pd.to_numeric(data.SibSp)
    data.Parch = pd.to_numeric(data.Parch)
    data.Fare = pd.to_numeric(data.Fare)

    # Filling null Values

    data.Age.fillna(data.Age.mean(), inplace=True)
    # One Hot Values
    oh_sex =  pd.get_dummies(data.Sex)
    data['IsMale'] = oh_sex['male']
    data['IsFemale'] = oh_sex['female']

    oh_embarked = pd.get_dummies(data.Embarked)
    data['Cherbourg'] = oh_embarked['C']
    data['Queenstown'] = oh_embarked['Q']
    data['Southampton'] = oh_embarked['S']

    # Drop unneeded features
    data.drop('Sex', axis=1, inplace=True)
    data.drop('Ticket', axis=1, inplace=True)
    data.drop('Embarked', axis=1,  inplace=True)
    data.drop('Name', axis=1, inplace=True)

    #Place fare into quartiles.
    fare_binned = []
    std = data.Fare.std()
    for fare in data.Fare.values:
        if fare <= std:
            fare_out = std
        elif (fare > std) and (fare <= std*2):
            fare_out = std*2
        elif (fare > std*2) and (fare <= std*3):
            fare_out = std*3
        elif fare > std*3:
            fare_out = std*4


        fare_binned.append(fare_out)

    data['FareBinned'] = fare_binned


    stripped_cabin = strip_to_first(data.Cabin)
    numeric_cabin = []
    for cabin in stripped_cabin.values:
        if cabin == 'A':
            cabin_out = 8
        elif cabin == 'B':
            cabin_out = 7
        elif cabin == 'C':
            cabin_out = 6
        elif cabin == 'D':
            cabin_out = 5
        elif cabin == 'E':
            cabin_out = 4
        elif cabin == 'F':
            cabin_out = 3
        elif cabin == 'G':
            cabin_out = 2
        elif cabin == 'T':
            cabin_out = 1
        else:
            cabin_out = 0
        numeric_cabin.append(cabin_out)
    data['Cabin'] = numeric_cabin

    data.rename(columns={'Cabin':'Deck'}, inplace=True)

    return data
