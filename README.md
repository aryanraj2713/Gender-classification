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
