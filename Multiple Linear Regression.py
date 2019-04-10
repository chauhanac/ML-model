#!/usr/bin/env python
# coding: utf-8

# In[69]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as pt


# In[70]:


data= pd.read_csv("50_startups.csv")


# In[71]:


data


# In[72]:


x=data.iloc[:,:-1].values


# In[73]:


y=data.iloc[:,4].values


# In[74]:


x


# In[75]:


y


# In[76]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# In[77]:


label = LabelEncoder() 


# In[78]:


x[:,3]= label.fit_transform(x[:,3])


# In[79]:


onehot = OneHotEncoder(categorical_features= [3])


# In[80]:


x= onehot.fit_transform(x).toarray()


# In[81]:


x= x[:,1:]


# In[82]:


x


# In[ ]:




from sklearn.model_selection import train_test_split 
# In[105]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)


# In[106]:


from sklearn.linear_model import LinearRegression


# In[107]:


regressor =LinearRegression()


# In[108]:


regressor.fit(x_train,y_train)


# In[109]:


y_pred= regressor.predict(x_test)


# In[110]:


y_pred


# In[111]:


from sklearn import metrics


# In[112]:


acc=(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[113]:


print("The accuracy is ",acc/100, "%")

