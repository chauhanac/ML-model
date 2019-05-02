#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as nm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


import warnings
warnings.filterwarnings("ignore")


# In[8]:


df= pd.read_csv("Adult_Income.csv")


# In[9]:


df.info()


# In[10]:


df_1  = df[df.workclass == "?"]


# In[11]:


df_1.info()


# In[12]:


df= df[df['workclass'] != '?']


# In[13]:


df.head()


# In[14]:


df_cat= df.select_dtypes(include=['object'])
df_cat.apply(lambda x: x=='?',axis=0).sum()


# In[15]:


df= df[df['occupation'] != '?']
df= df[df['native-country'] != '?']


# In[16]:


df.info()


# In[17]:


from sklearn import preprocessing


# In[18]:


df_cat= df.select_dtypes(include=['object'])


# In[19]:


le = preprocessing.LabelEncoder()


# In[20]:


df_cat=df_cat.apply(le.fit_transform)
df_cat.head()


# In[21]:


df= df.drop(df_cat.columns, axis=1)
df= pd.concat([df,df_cat], axis=1)


# In[22]:


df.info()


# In[23]:


df.head()


# In[24]:


from sklearn.model_selection import train_test_split


# In[25]:


x= df.drop('income', axis=1)
y=df['income']


# In[26]:


x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.3)


# In[27]:


from sklearn.tree import DecisionTreeClassifier


# In[28]:


dt_default = DecisionTreeClassifier(max_depth=5)
dt_default.fit(x_train, y_train)


# In[29]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[30]:


y_pred = dt_default.predict(x_test)


# In[31]:


print(classification_report(y_test,y_pred))


# In[32]:


print(accuracy_score(y_test,y_pred)*100)


# In[33]:


from IPython.display import Image
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
import pydotplus, graphviz


# In[34]:


features= list(df.columns[1:])
features


# In[35]:


import os
os.environ["PATH"] += os.pathsep + 'C:\Program Files (x86)\Graphviz2.38\bin'


# In[36]:


dot_data= StringIO()
export_graphviz(dt_default, out_file=dot_data, 
                feature_names=features, filled=True, rounded=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())


# In[37]:


graph.write_pdf("iris.pdf")


# In[ ]:




