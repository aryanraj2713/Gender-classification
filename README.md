# Gender-classification

*This is a ML model to classify Male and Females using some physical characterstics Data.*
*Python Libraries like Pandas,Numpy and Sklearn are used In this.*

Data set credits: Kraggle.com

**Importing Libraries**

```
import pandas as pd
import sklearn
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import VotingClassifier,RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

```
**Loading Data and exploring data**
```
data = pd.read_csv("L:\Gender classification\gender_classification_v7.csv")
data.head(20)
#checking for null values
data.isnull().sum()
data.describe()
```
**Encoding data and splitting data**
```
twogender = {'Female':0, 'Male':1}
data['gender'] = data['gender'].map(twogender)

X = data.drop('gender', axis=1)
y = data['gender']

#splitting data for testing and traing process
from sklearn.model_selection import train_test_split, GridSearchCV
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)
print(X_train.shape)
print(X_val.shape)
print(y_train.shape)
print(y_val.shape)
```
**Now we will test diffrent Sklearn Models to find best accuracy**

**Importing All required prerequisites**

```
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier
```
**Decision tree-classifier**
```
dt = DecisionTreeClassifier(random_state=0)

dt.fit(X_train, y_train)
dt_pred = dt.predict(X_val)
dt_acc = accuracy_score(y_val, dt_pred)
print('Accuracy of Decision Tree is: {:.2f}%'.format(dt_acc*100))
```
**randomforestClassifier**
```
rf = RandomForestClassifier(random_state=0)

rf.fit(X_train, y_train)
rf_pred = rf.predict(X_val)
rf_acc = accuracy_score(y_val, rf_pred)
print('Accuracy of Random Forest is: {:.2f}%'.format(rf_acc*100))
```
