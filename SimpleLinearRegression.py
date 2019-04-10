import numpy as np
import pandas as pd
import matplotlib.pyplot as pt


data = pd.read_csv("SabSal.csv")

x = data.iloc[:,:-1].values


y = data.iloc[:,1].values

from sklearn.model_selection import train_test_split


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1/3)


from sklearn.linear_model import LinearRegression


reg = LinearRegression() #object of linearRegression package


reg.fit(x_train, y_train)

y_pred = reg.predict(x_test)

pt.scatter(x_train,y_train,color = 'red')
pt.plot(x_train, reg.predict(x_train), color = 'blue')
pt.title("Salary vs. Experience (Train)")
pt.xlabel('Exp in years')
pt.ylabel('Salary in Rs.')
pt.show()                                            # PLOT TO SHOW THE MODEL ON TRAINING SET


pt.scatter(x_test,y_test,color = 'red')
pt.plot(x_test, reg.predict(x_test), color = 'blue')
pt.title("Salary vs. Experience (Test)")
pt.xlabel('Exp in years')
pt.ylabel('Salary in Rs.')
pt.show()                                           #PLOT TO SHOW THE FIT OF THE MODEL ON THE TESTING SET
