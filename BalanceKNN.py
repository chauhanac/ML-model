import numpy as nm
import pandas as pd
import matplotlib.pyplot as mp

data = pd.read_csv("BalanceScale.csv")

y=data.iloc[:,0].values

x=data.iloc[:,[1,2,3,4]].values

import sklearn
from sklearn.model_selection import train_test_split

x_train, x_test, y_train,y_test = train_test_split(x,y,test_size=0.30)


from sklearn.preprocessing import StandardScaler


sc=StandardScaler()


x_train=sc.fit_transform(x_train)

x_test=sc.transform(x_test)
from sklearn.neighbors import KNeighborsClassifier


classifier= KNeighborsClassifier(n_neighbors=7, metric='minkowski')

classifier.fit(x_train,y_train)

y_pred= classifier.predict(x_test)

from sklearn.metrics import confusion_matrix

c=confusion_matrix(y_test,y_pred)
