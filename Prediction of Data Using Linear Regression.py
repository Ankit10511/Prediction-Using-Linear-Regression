#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[12]:


dfx = pd.read_csv('Linear_X_Train.csv')
dfy = pd.read_csv('Linear_Y_Train.csv')

dfx = dfx.values
dfy = dfy.values

x = dfx.reshape((-1,1))
y = dfy.reshape((-1,1))
print(dfx.shape, dfy.shape)


# In[7]:


x = (x-x.mean())/x.std()
y = y
plt.scatter(x,y)
plt.show()


# In[9]:


from sklearn.linear_model import LinearRegression


# In[15]:


model = LinearRegression()


# In[16]:


model.fit(x,y)


# In[17]:


output = model.predict(x)


# In[18]:


bias = model.intercept_
coeff = model.coef_

print(bias)
print(coeff)


# In[19]:


model.score(x,y)


# In[24]:


plt.scatter(x,y,label='data')
plt.plot(x,output,color='black', label='prediction')
plt.legend()
plt.show()


# In[25]:


from collections import OrderedDict


# In[35]:


test_data = pd.read_csv('Linear_X_Test.csv')
test_data = test_data.values
x = test_data.reshape((-1,1))
y_predict = model.predict(test_data)
y_predict


# In[ ]:




