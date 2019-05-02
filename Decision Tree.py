import numpy as nm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings("ignore")

df= pd.read_csv("Adult_Income.csv")

df.info()

df_1  = df[df.workclass == "?"]

df_1.info()
df= df[df['workclass'] != '?']

df.head()

df_cat= df.select_dtypes(include=['object'])
df_cat.apply(lambda x: x=='?',axis=0).sum()


df= df[df['occupation'] != '?']
df= df[df['native-country'] != '?']


df.info()

from sklearn import preprocessing

df_cat= df.select_dtypes(include=['object'])

le = preprocessing.LabelEncoder()

df_cat=df_cat.apply(le.fit_transform)
df_cat.head()

df= df.drop(df_cat.columns, axis=1)
df= pd.concat([df,df_cat], axis=1)

df.info()


df.head()

from sklearn.model_selection import train_test_split

x= df.drop('income', axis=1)
y=df['income']

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.3)

from sklearn.tree import DecisionTreeClassifier

dt_default = DecisionTreeClassifier(max_depth=5)
dt_default.fit(x_train, y_train)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

y_pred = dt_default.predict(x_test)

print(classification_report(y_test,y_pred))

print(accuracy_score(y_test,y_pred)*100)

from IPython.display import Image
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
import pydotplus, graphviz

features= list(df.columns[1:])
features

import os
os.environ["PATH"] += os.pathsep + 'C:\Program Files (x86)\Graphviz2.38\bin'


dot_data= StringIO()
export_graphviz(dt_default, out_file=dot_data, 
                feature_names=features, filled=True, rounded=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())


graph.write_pdf("DT.pdf")
