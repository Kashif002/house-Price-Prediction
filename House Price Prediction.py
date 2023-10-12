#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.simplefilter("ignore")


# # Importing Dataset

# In[2]:


# Load the dataset from CSV
HouseDF=pd.read_csv('USA_housing.csv')


# In[3]:


# Exploratory Data Analysis (EDA)
# Let's take a quick look at the first few rows of the dataset
HouseDF.head()


# In[4]:


# Let's take a quick look at the last few rows of the dataset
HouseDF.tail()


# In[5]:


# Summary statistics of the datase
print(HouseDF.describe())


# In[6]:


# Check for missing values
print(HouseDF.isnull().sum())


# # Data Visualization

# In[7]:


# Correlation matrix to understand feature relationships
correlation_matrix = HouseDF.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()


# # Data Preparation

# In[8]:


# Preprocessing: Selecting features and target variable
X = HouseDF[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition']]
y = HouseDF['price']


# In[9]:


# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[13]:


X_test.head()


# In[15]:


y_test.head()


# # Training Model

# In[10]:


# Building the Linear Regression Model
model = LinearRegression()

# Fitting the model on the training data
model.fit(X_train, y_train)


# In[11]:


# Model Evaluation
y_pred = model.predict(X_test)

# Mean Squared Error and R-squared for model evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)


# # Prediction

# In[12]:


# Predictions and Visualization
# To visualize the predictions against actual prices, we'll use a scatter plot
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices vs. Predicted Prices")
plt.show()

# We can also create a residual plot to check the model's performance
residuals = y_test - y_pred
plt.scatter(y_test, residuals)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Actual Prices")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

# Lastly, let's use the trained model to make predictions on new data and visualize the results
new_data = [[4,2,2090,6630,1,0,0,3]]
predicted_price = model.predict(new_data)

print("Predicted Price:", predicted_price[0])

