#!/usr/bin/env python
# coding: utf-8

# In[50]:


import numpy as nm


# In[51]:


import pandas as pd
import matplotlib.pyplot as mp


# In[52]:


data = pd.read_csv("BalanceScale.csv")


# In[53]:


y=data.iloc[:,0].values


# In[54]:


y


# In[55]:


x=data.iloc[:,[1,2,3,4]].values


# In[56]:


x


# In[57]:


import sklearn


# In[58]:


from sklearn.model_selection import train_test_split


# In[103]:


x_train, x_test, y_train,y_test = train_test_split(x,y,test_size=0.30)


# In[104]:


from sklearn.preprocessing import StandardScaler


# In[105]:


sc=StandardScaler()


# In[106]:


x_train=sc.fit_transform(x_train)


# In[107]:


x_test=sc.transform(x_test)


# In[110]:


from sklearn.neighbors import KNeighborsClassifier


# In[111]:


classifier= KNeighborsClassifier(n_neighbors=7, metric='minkowski')


# In[112]:


classifier.fit(x_train,y_train)


# In[113]:


y_pred= classifier.predict(x_test)


# In[114]:


from sklearn.metrics import confusion_matrix


# In[115]:


c=confusion_matrix(y_test,y_pred)


# In[116]:


c


# In[117]:


acc=(c[0][0]+c[1][1]+c[2][2])/sum(map(sum,c))*100


# In[118]:


print("The model will predict correct balance with", acc,"% accuracy")


# In[ ]:




