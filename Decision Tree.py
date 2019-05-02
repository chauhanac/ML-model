#!/usr/bin/env python
# coding: utf-8

# In[172]:


import numpy as nm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[173]:


import warnings
warnings.filterwarnings("ignore")


# In[174]:


df= pd.read_csv("Adult_Income.csv")


# In[175]:


df.info()


# In[176]:


df_1  = df[df.workclass == "?"]


# In[177]:


df_1.info()


# In[178]:


df= df[df['workclass'] != '?']


# In[179]:


df.head()


# In[180]:


df_cat= df.select_dtypes(include=['object'])
df_cat.apply(lambda x: x=='?',axis=0).sum()


# In[181]:


df= df[df['occupation'] != '?']
df= df[df['native-country'] != '?']


# In[182]:


df.info()


# In[183]:


from sklearn import preprocessing


# In[184]:


df_cat= df.select_dtypes(include=['object'])


# In[185]:


le = preprocessing.LabelEncoder()


# In[186]:


df_cat=df_cat.apply(le.fit_transform)
df_cat.head()


# In[187]:


df= df.drop(df_cat.columns, axis=1)
df= pd.concat([df,df_cat], axis=1)


# In[188]:


df.info()


# In[189]:


df.head()


# In[190]:


from sklearn.model_selection import train_test_split


# In[191]:


x= df.drop('income', axis=1)
y=df['income']


# In[192]:


x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.3)


# In[193]:


from sklearn.tree import DecisionTreeClassifier


# In[194]:


dt_default = DecisionTreeClassifier(max_depth=5)
dt_default.fit(x_train, y_train)


# In[195]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[196]:


y_pred = dt_default.predict(x_test)


# In[197]:


print(classification_report(y_test,y_pred))


# In[198]:


print(accuracy_score(y_test,y_pred)*100)


# In[199]:


from IPython.display import Image
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
import pydotplus, graphviz


# In[200]:


features= list(df.columns[1:])
features


# In[201]:


import os
os.environ["PATH"] += os.pathsep + 'C:\Program Files (x86)\Graphviz2.38\bin'


# In[202]:


dot_data= StringIO()
export_graphviz(dt_default, out_file=dot_data, 
                feature_names=features, filled=True, rounded=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())


# In[203]:


graph.write_pdf("iris.pdf")


# In[204]:


#tuning with MAX DEPTH


# In[205]:


from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV


# In[206]:


n_fold=5
parameter= {'max_depth': range(1,40)}
dtree = DecisionTreeClassifier(criterion= 'gini')
tree = GridSearchCV(dtree,parameter,
                    cv=n_fold,
                    scoring= "accuracy")
tree.fit(x_train,y_train)


# In[207]:


scores = tree.cv_results_
pd.DataFrame(scores).head()


# In[208]:


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


# In[209]:


#As it is overfitting
#tuning with min_sammple_leaf


# In[210]:


n_folds=5
parameters= {'min_samples_leaf': range(5,200,20)}
dtree = DecisionTreeClassifier(criterion= 'gini')
tree = GridSearchCV(dtree,parameters,
                    cv=n_folds,
                    scoring= "accuracy")
tree.fit(x_train,y_train)


# In[211]:


scores = tree.cv_results_
pd.DataFrame(scores).head()


# In[212]:


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


# In[213]:


#tuning with minimum sample split


# In[214]:


n_folds=5
parameters= {'min_samples_split': range(5,200,20)}
dtree = DecisionTreeClassifier(criterion= 'gini')
tree = GridSearchCV(dtree,parameters,
                    cv=n_folds,
                    scoring= "accuracy")
tree.fit(x_train,y_train)


# In[215]:


scores = tree.cv_results_
pd.DataFrame(scores).head()


# In[216]:


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


# In[217]:


param_grid = {"max_depth": range(1,150,5),
              "min_samples_split": range(2,150,50),
              "min_samples_leaf": range(1,150,50),
              'criterion':["entropy","gini"]
    
}


# In[218]:


n_folds=5
dtree= DecisionTreeClassifier()


# In[219]:


grid_search= GridSearchCV(estimator = dtree, param_grid= param_grid,
                         cv=n_folds, verbose=1)


# In[220]:


grid_search.fit(x_train,y_train)


# In[221]:


cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results.head()


# In[222]:


print("Best Accuracy",grid_search.best_score_*100)


# In[223]:


print("Best Estimator",grid_search.best_estimator_)


# In[235]:


best_vals= DecisionTreeClassifier(criterion='gini',
                                 max_depth=3,
                                 min_samples_leaf=1,
                                 min_samples_split=102)                #max_depth not optimal as it becomes too complicated
best_vals.fit(x_train, y_train)


# In[236]:


best_vals.score(x_test, y_test)*100


# In[237]:


dot_data = StringIO()
export_graphviz(best_vals, out_file= dot_data,
               feature_names=features, filled=True, rounded= True)


# In[238]:


graph= pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())

