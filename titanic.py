import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV

# DataFrame
df = pd.read_csv('./data/train.csv')
df = df.fillna(df.mean())

# Functions to simplify columns
## Simplify Cabin feature
def simplify_Cabin(df):
    df.Cabin = df.Cabin.fillna('N')
    df.Cabin = df.Cabin.apply(lambda x: x[0])
    return df

## Drop a few features
def dropFeatures(df):
    df = df.drop(['Ticket', 'Embarked'], axis=1)
    return df

## Call Above Function
def Simplify(df):
    # df = simplify_Age(df)
    df = simplify_Cabin(df)
    df = dropFeatures(df)
    return df

# Call Simplify
df = Simplify(df)

#Define Features
features = ['Pclass', 'Name', 'Age', 'SibSp', 'Fare', 'Cabin', 'Sex']

# Encoding
for feature in features:
    enc = preprocessing.LabelEncoder()
    enc = enc.fit(df[feature])
    df[feature] = enc.transform(df[feature])

# Features and Label Defining
X = df.drop(['Survived', 'PassengerId'], axis=1)
y = df.Survived

#Split values into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))