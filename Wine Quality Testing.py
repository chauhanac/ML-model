#!/usr/bin/env python
# coding: utf-8

# In[51]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[52]:


wine = pd.read_csv("winequality.csv")


# In[53]:


wine


# In[54]:


x=wine.iloc[:, wine.columns != 'quality' ].values


# In[55]:


x


# In[56]:


y=wine.iloc[:,11].values


# In[57]:


y


# In[58]:


from sklearn.model_selection import train_test_split


# In[109]:


x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.25,random_state =0)


# In[110]:


from sklearn.linear_model import LogisticRegression


# In[111]:


log_reg=LogisticRegression()


# In[112]:


log_reg.fit(x_train,y_train)


# In[113]:


y_pred= log_reg.predict(x_test)


# In[114]:


y_pred


# In[115]:


from sklearn import metrics


# In[116]:


acc=(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[117]:


acc


# In[120]:


print("From the given parameters the wine quality accuracy is ",acc*100, "%")

