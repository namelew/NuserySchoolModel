import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn import preprocessing as pp
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier as rfc

FILE="./nursery.csv"

# load the dataset

df=pd.read_csv(FILE, names=["parents", "has_nurs", "form", "children", "housing", "finance", "social", "health", "reputation"])

# encolding collumns

for collumn in df.columns:
  laben=pp.LabelEncoder()
  laben.fit(df[collumn])
  df[collumn]=laben.transform(df[collumn])

# spliting the data and select best parameters

Y = df['reputation']
X = df.drop(['reputation'], axis=1)

X = SelectKBest(chi2, k=8).fit_transform(X, Y)

X_train, X_test, y_train, y_test = tts(X,Y, test_size=0.2, random_state=42)

print(X,Y)
# training model