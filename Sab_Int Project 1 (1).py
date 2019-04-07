#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as nm
import pandas as pd
import matplotlib.pyplot as mp


# In[6]:


dataset = pd.read_csv("Social_Network_Ads.csv")


# In[7]:


dataset


# In[115]:


x=dataset.iloc[:,[2,3]].values


# In[116]:


y=dataset.iloc[:,[4]].values


# In[117]:


import sklearn


# In[118]:


from sklearn.cross_validation import train_test_split


# In[153]:


x_train, x_test, y_train,y_test = train_test_split(x,y,test_size=0.25)


# In[154]:


from sklearn.preprocessing import StandardScaler


# In[155]:


sc=StandardScaler()


# In[156]:


x_train= sc.fit_transform(x_train)


# In[157]:


x_test= sc.transform(x_test)


# In[144]:


from sklearn.neighbors import KNeighborsClassifier


# In[159]:


classifier = KNeighborsClassifier(n_neighbors= 7, metric = 'minkowski')


# In[160]:


classifier.fit(x_train, y_train)


# In[161]:


y_pred= classifier.predict(x_test)


# In[162]:


from sklearn.metrics import confusion_matrix


# In[163]:


cm= confusion_matrix(y_test,y_pred)


# In[164]:


cm


# In[166]:


accuracy=(94/100)*100


# In[167]:


print("The model predicts values by",accuracy,"% accuracy")


# In[ ]:




