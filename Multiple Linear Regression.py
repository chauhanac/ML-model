import numpy as np
import pandas as pd
import matplotlib.pyplot as pt


data= pd.read_csv("50_startups.csv")


x=data.iloc[:,:-1].values

y=data.iloc[:,4].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder


label = LabelEncoder() 


x[:,3]= label.fit_transform(x[:,3])

onehot = OneHotEncoder(categorical_features= [3])

x= onehot.fit_transform(x).toarray()


x= x[:,1:]


from sklearn.model_selection import train_test_split 

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)


from sklearn.linear_model import LinearRegression


regressor =LinearRegression()

regressor.fit(x_train,y_train)


y_pred= regressor.predict(x_test)


from sklearn import metrics


acc=(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


print("The accuracy is ",acc/100, "%")
