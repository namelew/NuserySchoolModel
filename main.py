import pandas as pd
import sys
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing as pp
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.metrics import recall_score as recall

FILE="./nursery.csv"
CASE="bared"
KBEST=8
KFOLD=7
RCAVERAGE="micro"

# load args

if len(sys.argv) > 1:
  CASE = sys.argv[1]

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

if CASE in ["kbest", "tbest", "hpbest"]:
  X = SelectKBest(chi2, k=KBEST).fit_transform(X, Y)

if CASE != "tbest":
  X_train, X_test, y_train, y_test = tts(X,Y, test_size=0.2, random_state=42)

# training model and evaluate results
if CASE == "bared":
  forest = rfc()
  forest.fit(X_train, y_train)
  y_hat = forest.predict(X_test)
  print(f"No optimized model\n\tprecision: {forest.score(X_test, y_test)}\n\trecall: {recall(y_hat, y_test, average=RCAVERAGE)}")
if CASE == "kbest":
  forest = rfc()
  forest.fit(X_train, y_train)
  y_hat = forest.predict(X_test)
  print(f"Optimized model with Select-{KBEST}-Best\n\tprecision: {forest.score(X_test, y_test)}\n\trecall: {recall(y_hat, y_test, average=RCAVERAGE)}")
if CASE == "tbest":
  kf = KFold(n_splits=KFOLD, random_state=42, shuffle=True)
  forest = rfc()

  accs = []
  recalls = []

  for train_index, test_index in kf.split(X):
      X_train, X_test = X[train_index, :], X[test_index, :]
      y_train, y_test = Y[train_index], Y[test_index]
      forest.fit(X_train, y_train)
      y_hat = forest.predict(X_test)
      accs.append(forest.score(X_test, y_test))
      recalls.append(recall(y_hat, y_test, average=RCAVERAGE))
  accs.sort()
  recalls.sort()

  print(f"Optimized model with Select-{KBEST}-Best and Cross-Validation\n\tprecision: {accs[0]}\n\trecall: {recalls[0]}")
if CASE == "hpbest":
  parameters = {
      "n_estimators": [25, 50, 100, 200, 400],
      "criterion": ["gini", "entropy", "log_loss"],
      "max_depth": [5, 25, 50, 100, 200],
      "min_samples_split": [2, 4, 8, 16, 32],
      "min_samples_leaf": [1, 2, 3, 4, 5],
      "min_weight_fraction_leaf": [0.0, 0.1, 0.2, 0.3, 0.4],
      "max_features": ["sqrt", "log2"],
      "max_leaf_nodes": [10000, 20000, 30000, 40000, 50000],
      "min_impurity_decrease": [0.0, 0.01, 0.02, 0.03, 0.04],
      "random_state": [42],
  }

  forestGrid = GridSearchCV(rfc(), parameters)
  forestGrid.fit(X_train, y_train)
  forest = rfc(forestGrid.best_estimator_)
  y_hat = forest.predict(X_test)
  print(f"Optimized model with Select-{KBEST}-Best and GridSearchCV\n\tprecision: {forest.score(X_test, y_test)}\n\trecall: {recall(y_hat, y_test, average=RCAVERAGE)}")