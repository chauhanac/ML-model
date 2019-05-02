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

#tuning with MAX DEPTH

from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

n_fold=5
parameter= {'max_depth': range(1,40)}
dtree = DecisionTreeClassifier(criterion= 'gini')
tree = GridSearchCV(dtree,parameter,
                    cv=n_fold,
                    scoring= "accuracy")
tree.fit(x_train,y_train)

scores = tree.cv_results_
pd.DataFrame(scores).head()

plt.figure()
plt.plot(scores["param_max_depth"],
        scores["mean_train_score"],
        label="training_accuracy")
plt.plot(scores["param_max_depth"],
        scores["mean_test_score"],
        label="test_accuracy")

plt.xlabel("max_depth")
plt.ylabel("accuracy")
plt.legend()
plt.show()


#Above model is overfitting so,
#tuning with min_sammple_leaf

n_folds=5
parameters= {'min_samples_leaf': range(5,200,20)}
dtree = DecisionTreeClassifier(criterion= 'gini')
tree = GridSearchCV(dtree,parameters,
                    cv=n_folds,
                    scoring= "accuracy")
tree.fit(x_train,y_train)

scores = tree.cv_results_
pd.DataFrame(scores).head()

plt.figure()
plt.plot(scores["param_min_samples_leaf"],
        scores["mean_train_score"],
        label="training_accuracy")
plt.plot(scores["param_min_samples_leaf"],
        scores["mean_test_score"],
        label="test_accuracy")

plt.xlabel("min_samples_leaf")
plt.ylabel("accuracy")
plt.legend()
plt.show()


#tuning with minimum sample split


n_folds=5
parameters= {'min_samples_split': range(5,200,20)}
dtree = DecisionTreeClassifier(criterion= 'gini')
tree = GridSearchCV(dtree,parameters,
                    cv=n_folds,
                    scoring= "accuracy")
tree.fit(x_train,y_train)

scores = tree.cv_results_
pd.DataFrame(scores).head()

plt.figure()
plt.plot(scores["param_min_samples_split"],
        scores["mean_train_score"],
        label="training_accuracy")
plt.plot(scores["param_min_samples_split"],
        scores["mean_test_score"],
        label="test_accuracy")

plt.xlabel("min_samples_split")
plt.ylabel("accuracy")
plt.legend()
plt.show()


param_grid = {"max_depth": range(1,150,5),
              "min_samples_split": range(2,150,50),
              "min_samples_leaf": range(1,150,50),
              'criterion':["entropy","gini"]
             }

n_folds=5
dtree= DecisionTreeClassifier()

grid_search= GridSearchCV(estimator = dtree, param_grid= param_grid,cv=n_folds, verbose=1)

grid_search.fit(x_train,y_train)

cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results.head()

print("Best Accuracy",grid_search.best_score_*100)

print("Best Estimator",grid_search.best_estimator_)

best_vals= DecisionTreeClassifier(criterion='gini', max_depth=3, min_samples_leaf=1, min_samples_split=102)                
#max_depth not optimal as decision tree becomes too complicated

best_vals.fit(x_train, y_train)

best_vals.score(x_test, y_test)*100

dot_data = StringIO()
export_graphviz(best_vals, out_file= dot_data,
               feature_names=features, filled=True, rounded= True)


graph= pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())
