import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing as pp
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.metrics import recall_score as recall
from imblearn.over_sampling import SMOTE

FILE="./nursery.csv"
KFOLD=4
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

X,Y = SMOTE(k_neighbors=1).fit_resample(X,Y)

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

# training model and evaluate results
forest = rfc(n_jobs=-1,random_state=42,)

forest.fit(X_train, y_train)
y_hat = forest.predict(X_test)
print(f"Model Metrics\n\tprecision: {forest.score(X_test, y_test)}\n\trecall: {recall(y_hat, y_test, average=RCAVERAGE)}")