#!/usr/bin/env python
# coding: utf-8

# ## Aishwarya Walkar
# ### Data Science and Business Analytics Intern @ The Sparks Foundation
# #### Topic : Exploratory Data Analysis (EDA) - Simple linear regression 
# #### Dataset : http://bit.ly/w-data

# In[8]:


# Importing all libraries required in this notebook
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.datasets import make_blobs


# In[ ]:





# In[69]:


#Import the data
url="http://bit.ly/w-data"
data=pd.read_csv(url)
print("The data is imported successfully")
data


# In[70]:



data.describe()


# In[71]:


data.isnull().sum()


# #DATA VISUALIZATION
# Now let's plot a graph of our data so that it will give us clear idea about data.

# In[72]:


#Plotting the distribution of scores
data.plot(x='Hours', y='Scores', style='1')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# In[73]:


#Splitting training and testing data
x=data.iloc[:,:-1].values
y=data.iloc[:,1].values


# In[74]:


from sklearn.preprocessing import StandardScaler
sc =StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
x_train, x_test, y_train, y_test= train_test_split(x, y,test_size=0.20,random_state=0)


# In[75]:


x_train.shape


# In[76]:


y_train.shape


# In[88]:


#Training the model
from sklearn.linear_model import LinearRegression
linearRegressor= LinearRegression()
linearRegressor.fit(x_train, y_train)
y_predict= linearRegressor.predict(x_test)


# In[89]:


#Training the algorithm
#Now the spliting of our data into training and testing sets is done, now it's time to train our algorithm.

regressor = LinearRegression()  
regressor.fit(x_train, y_train) 

print("Training complete.")


# In[90]:


# Plotting the regression line
line = regressor.coef_*x+regressor.intercept_
# Plotting for the test data
plt.scatter(x, y)
plt.plot(x, line);
plt.show()


# #Checking the accuracy scores for training and test set
# 

# In[91]:


print('Test Score')
print(regressor.score(x_test, y_test))
print('Training Score')
print(regressor.score(x_train, y_train))


# #Now we make predictions

# In[92]:


'''a = {'actual': y_test,'predicted': y_predict}
df = pd.DataFrame.from_dict(a,orient='index')
df.transpose()

print(df)
'''


# In[93]:


data.shape


# In[97]:


data= pd.DataFrame(columns={'Actual': y_test,'Predicted': y_predict})
data


# In[98]:


#Let's predict the score for 9.25 hpurs
print('Score of student who studied for 9.25 hours a dat', regressor.predict([[9.25]]))


# In[99]:


#Checking the efficiency of model
mean_squ_error = mean_squared_error(y_test, y_predict)
mean_abs_error = mean_absolute_error(y_test, y_predict)
print("Mean Squred Error:",mean_squ_error)
print("Mean absolute Error:",mean_abs_error)


# In[100]:


y_predict


# In[101]:


y_test


# In[ ]:




