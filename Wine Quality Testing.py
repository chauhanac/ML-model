import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

wine = pd.read_csv("winequality.csv")

wine

x=wine.iloc[:, wine.columns != 'quality' ].values

y=wine.iloc[:,11].values

from sklearn.model_selection import train_test_split


x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.25,random_state =0)

from sklearn.linear_model import LogisticRegression


log_reg=LogisticRegression()


log_reg.fit(x_train,y_train)


y_pred= log_reg.predict(x_test)


from sklearn import metrics

acc=(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


print("From the given parameters the wine quality accuracy is ",acc*100, "%")

