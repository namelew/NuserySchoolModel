import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing as pp
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.metrics import recall_score as recall

FILE="./nursery.csv"
KFOLD=7
RCAVERAGE="micro"

# load the dataset

df=pd.read_csv(FILE, names=["parents", "has_nurs", "form", "children", "housing", "finance", "social", "health", "reputation"])

# dumping collumns
for collumn in df.columns:
  if len(df[collumn].unique()) > 2 and collumn not in ["reputation", "children"]:
    dump=pd.get_dummies(df[collumn],drop_first=True)
    df = pd.concat([df,dump],axis=1) # add new columns
    df.drop([collumn],inplace=True,axis=1) # delete the original attribute

# encolding collumns
for collumn in df.columns:
  laben=pp.LabelEncoder()
  laben.fit(df[collumn])
  df[collumn]=laben.transform(df[collumn])

# spliting the data and select best parameters

Y = df['reputation']
X = df.drop(['reputation'], axis=1)

kf = KFold(n_splits=KFOLD, random_state=42, shuffle=True)
forest = rfc()

samples = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X.values[train_index], X.values[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    forest.fit(X_train, y_train)
    samples.append({
      "train": (X.values[train_index], Y[train_index]),
      "test": (X.values[test_index], Y[test_index]),
      "score": forest.score(X_test, y_test)
    })

samples.sort(key=lambda x:x["score"], reverse=True)

if KFOLD%2 == 1:
  best = round(KFOLD%2)
else:
  best = round(KFOLD%2) + 1

X_train = samples[best]["train"][0]
y_train = samples[best]["train"][1]
X_test = samples[best]["test"][0] 
y_test = samples[best]["test"][1]

# find the best hyperparameters

parameters = {
      "n_estimators": [25, 50, 100, 200, 400, 800, 1600, 2400, 4800],
      "criterion": ["gini", "entropy", "log_loss"],
      "max_depth": [1, 2, 4, 5, 25, 50, 100, 200],
      "min_samples_split": [2, 4, 8, 16, 32, 64, 128],
      "min_samples_leaf": [1, 2, 3, 4, 5, 6, 8, 9, 10],
      "min_weight_fraction_leaf": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 1],
      "max_features": ["sqrt", "log2"],
      "max_leaf_nodes": [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000],
      "min_impurity_decrease": [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
      "warm_start": [True, False],
      "oob_score": [True, False],
      "n_jobs": [-1],
      "random_state": [42],
  }

forestGrid = GridSearchCV(rfc(), parameters)
forestGrid.fit(X_train, y_train)

df = pd.DataFrame(forestGrid.cv_results_)
df.to_csv('grid_search_results_forest.csv', index=False)

# training model and evaluate results
forest = forestGrid.best_estimator_
forest.fit(X_train, y_train)
y_hat = forest.predict(X_test)
print(f"Model Metrics\n\tprecision: {forest.score(X_test, y_test)}\n\trecall: {recall(y_hat, y_test, average=RCAVERAGE)}")