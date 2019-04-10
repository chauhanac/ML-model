#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as pt


# In[5]:


data = pd.read_csv("SabSal.csv")


# In[6]:


data


# In[30]:


x = data.iloc[:,:-1].values


# In[31]:


x


# In[32]:


y = data.iloc[:,1].values


# In[33]:


y


# In[34]:


from sklearn.model_selection import train_test_split


# In[35]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1/3)


# In[36]:


from sklearn.linear_model import LinearRegression


# In[37]:


reg = LinearRegression() #object of linearRegression package


# In[38]:


reg.fit(x_train, y_train)


# In[39]:


y_pred = reg.predict(x_test)


# In[40]:


y_pred


# In[41]:


y_test


# In[59]:


pt.scatter(x_train,y_train,color = 'red')
pt.plot(x_train, reg.predict(x_train), color = 'blue')
pt.title("Salary vs. Experience (Train)")
pt.xlabel('Exp in years')
pt.ylabel('Salary in Rs.')
pt.show()


# In[60]:


pt.scatter(x_test,y_test,color = 'red')
pt.plot(x_test, reg.predict(x_test), color = 'blue')
pt.title("Salary vs. Experience (Test)")
pt.xlabel('Exp in years')
pt.ylabel('Salary in Rs.')
pt.show()


# In[ ]:




