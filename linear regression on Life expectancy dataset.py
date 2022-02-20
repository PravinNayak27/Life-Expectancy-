#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np                  #importing the required libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


from warnings import filterwarnings                   #For ignoring warnings
filterwarnings('ignore')


# In[3]:


Data = pd.read_csv("C:/Users/Dell/Desktop/Life Expectancy Data.csv")        #importing dataset
Data.head()


# In[4]:


Data.shape                                                   # shape of the data


# In[5]:


Data.columns                                               


# In[6]:


Data.info()


# In[7]:


plt.figure(figsize=(20,5))               #checking for outliers using boxplot
Data.boxplot()
plt.xticks(rotation=90)                  #rotating variable names  
plt.show()


#most of outliers lies in population 


# In[8]:


Data_1=Data.drop(['Population'],axis=1)                                      #drop population as it contains outliers


# In[9]:


plt.figure(figsize=(20,5))               #checking for outliers using boxplot
Data_1.boxplot()
plt.xticks(rotation=90)                  #rotating variable names  
plt.show()


# In[10]:


Data_1=Data_1.drop(['Country','Status','Year'],axis=1)                     #dropping columns as it contains outliers


# In[11]:


sns.pairplot(Data_1,size=2,aspect=1.5)                                    #pairplot 
plt.show()                                                                        


# In[12]:


Data_1.corr()                                                      


# In[13]:


plt.figure(figsize=(15,10))                                               #heatmop to see correlation


sns.heatmap(Data_1.corr(),annot=True)
plt.show()


# In[14]:


from sklearn.preprocessing import StandardScaler                        #For standardizing the data
scaler = StandardScaler().fit(Data_1)
Data_1=scaler.transform(Data_1)


# In[15]:


Data_df=pd.DataFrame(Data_1)
Data_df.head()


# In[16]:


Data_df.columns = ['Life expectancy ', 'Adult Mortality',                                      #rreplacing column indexes with their names
                   
       'infant deaths', 'Alcohol', 'percentage expenditure', 'Hepatitis B',
       'Measles ', ' BMI ', 'under-five deaths ', 'Polio', 'Total expenditure',
       'Diphtheria ', ' HIV/AIDS', 'GDP',' thinness  1-19 years', ' thinness 5-9 years',
       'Income composition of resources', 'Schooling']
Data_df.head()


# In[17]:


Data_df.info()                                                      


# In[18]:


Data_df.replace([np.inf, -np.inf], np.nan                               #replacing all the infinite values
Data_df.dropna(inplace=True)


# In[19]:


import statsmodels.api as sm                                                  
X = Data_df.iloc[:,1:]
Y = Data_df.iloc[:,0]


# In[20]:


print(X)
print(X.shape)


# In[21]:


print(Y)
print(Y.shape)


# In[22]:


from sklearn.model_selection import train_test_split    #importing train_test_split for splitting the data in train and test model

def split(X,Y):
    return train_test_split(X,Y, test_size=0.3, random_state=10)


# In[23]:


X_train,X_test,Y_train,Y_test = split(X,Y)
print("Train cases are given below :")
print(X_train.shape)
print( Y_train.shape)
print("\nTest cases are given below :")
print(X_test.shape)
print(Y_test.shape)


# In[24]:


from sklearn.linear_model import LinearRegression                 #Importing required libraries

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# In[25]:


model = LinearRegression().fit(X_train, Y_train)                  # Fitting the regression model
y_pred_test= model.predict(X_test)                                # predicting test model


# In[26]:


print('Intercept :',model.intercept_)                             #intercept & coefficients
print('Coefficients :',model.coef_)


# In[27]:


y_pred_train=model.predict(X_train)                              #predicting train model
print('Intercepts :',model.intercept_)                           #intercept & coefficients
print('Coefficients :',model.coef_)


# In[28]:


from sklearn import metrics


# In[29]:


MAE_train = metrics.mean_absolute_error(Y_train,y_pred_train)                   #Mean absolute Error for train data
print('Mean absolute error for train data is {}'.format(MAE_train))

MAE_test = metrics.mean_absolute_error(Y_test,y_pred_test)                      #Mean absolute Error for test data
print('Mean absolute error for test data is {}'.format(MAE_test))


# In[30]:


MSE_train = metrics.mean_squared_error(Y_train,y_pred_train)                    #Mean Squared Error for train data
print('Mean Squared error for train data is {}'.format(MSE_train))

MSE_test = metrics.mean_squared_error(Y_test,y_pred_test)                       #Mean Squared Error for test data
print('Mean Squared error for test data is {}'.format(MSE_test))


# In[31]:


RMSE_train =np.sqrt(metrics.mean_squared_error(Y_train,y_pred_train))           # Root Mean Squared Error for train data
print('Root Mean Squared error for train data is {}'.format(RMSE_train))

RMSE_test = np.sqrt(metrics.mean_squared_error(Y_test,y_pred_test))             # Root Mean Squared Error for test data
print('Root Mean Squared error for test data is {}'.format(RMSE_test))


# In[32]:


yhat = model.predict(X_train)                                                   #calculating R-square and Adjusted R Square

SS_Residual = sum((Y_train-yhat)**2)

SS_Total = sum((Y_train-np.mean(Y_train))**2)

R_Square =1-(float(SS_Residual))/SS_Total

Adj_R_Square = 1-(1-R_Square)*(len(Y_train)-1)/(len(Y_train)-X_train.shape[1]-1)

print('R-Squared value is {}'.format(R_Square))
print('Adjusted R-Squared Value is {}'.format(Adj_R_Square))


# In[33]:


yhat = model.predict(X_test)                                                     #calculating R-square and Adjusted R Square

SS_Residual = sum((Y_test-yhat)**2)

SS_Total = sum((Y_test-np.mean(Y_test))**2)

R_Square =1-(float(SS_Residual))/SS_Total

Adj_R_Square = 1-(1-R_Square)*(len(Y_test)-1)/(len(Y_test)-X_test.shape[1]-1)

print('R-Squared value is {}'.format(R_Square))
print('Adjusted R-Squared Value is {}'.format(Adj_R_Square))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




