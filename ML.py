from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
balance_data=pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data",
                           sep=',', header=None)

le = preprocessing.LabelEncoder()
balance_data = balance_data.apply(le.fit_transform)

X = balance_data.values[:, 0:5]
Y = balance_data.values[:, 6]

train, test, train_labels, test_labels = train_test_split(X, Y, test_size=0.50, random_state=42)

gnb = GaussianNB()
model = gnb.fit(train, train_labels)

preds = gnb.predict(test)
print(preds)

# Evaluate accuracy
print(accuracy_score(test_labels, preds))