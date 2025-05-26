# -*- coding: utf-8 -*-
"""Machine Learning Project

Original file was from Google Colab,
now adapted for local environment.

# Boston House Pricing
"""

# Importing the dependencies
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline  # (Optional for Jupyter, can remove if running as .py script)

# Importing the dataset
# Removed: from google.colab import files
# Removed: uploaded = files.upload()

# Instead, make sure 'HousingData.csv' is in the same folder as this script
df = pd.read_csv('HousingData.csv')

# Checking the top 5 data from the dataset
print(df.head())

"""# Preparing the Dataset"""

# Checking the information of the dataset
print(df.info())

df.rename(columns={'MEDV': 'Price'}, inplace=True)

print(df.isnull().sum())

# Filling the missing values

# Fill continuous columns with median
for col in ['CRIM', 'ZN', 'INDUS', 'AGE', 'LSTAT']:
    df[col] = df[col].fillna(df[col].median())

# Fill CHAS (categorical) with mode
df['CHAS'] = df['CHAS'].fillna(df['CHAS'].mode()[0])

# Check for missing values
print(df.isnull().sum())

"""# Exploratory Data Analysis"""

# Correlation
print(df.corr())

sns.pairplot(df)
plt.show()

plt.scatter(df['CRIM'], df['Price'])
plt.xlabel("Crime Rate")
plt.ylabel("Price")
plt.show()

plt.scatter(df['RM'], df['Price'])
plt.xlabel("RM")
plt.ylabel("Price")
plt.show()

sns.regplot(x="RM", y="Price", data=df)
plt.show()

sns.regplot(x="LSTAT", y="Price", data=df)
plt.show()

sns.regplot(x="CHAS", y="Price", data=df)
plt.show()

sns.regplot(x="PTRATIO", y="Price", data=df)
plt.show()

"""# Splitting the data into dependent and independent data"""

X = df.drop(columns=["Price"], axis=1)
y = df["Price"]

print(X.head())
print(y.head())

# Splitting the data into train and test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(X_train.head())
print(X_test.head())

"""# Feature Scaling"""

# Standard scaler
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

import pickle

pickle.dump(scaler, open('scaling.pkl', 'wb'))

print(X_train)
print(X_test)

"""# Model Training"""

from sklearn.linear_model import LinearRegression

regression = LinearRegression()

regression.fit(X_train, y_train)

print(regression.coef_)
print(regression.intercept_)
print(regression.get_params())

reg_pred = regression.predict(X_test)

print(reg_pred)

"""# Assumptions"""

plt.scatter(y_test, reg_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.show()

residuals = y_test - reg_pred
print(residuals)

sns.displot(residuals, kind="kde")
plt.show()

plt.scatter(reg_pred, residuals)
plt.xlabel("Predicted Price")
plt.ylabel("Residuals")
plt.show()

from sklearn.metrics import mean_squared_error, mean_absolute_error

print("MAE:", mean_absolute_error(y_test, reg_pred))
print("MSE:", mean_squared_error(y_test, reg_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, reg_pred)))

"""# R Square and adjusted R square"""

from sklearn.metrics import r2_score

score = r2_score(y_test, reg_pred)
print("R2 Score:", score)

# Display adjusted R-squared
adjusted_r2 = 1 - (1 - score) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)
print("Adjusted R2 Score:", adjusted_r2)

"""# Pickling The Model file For Deployment"""

pickle.dump(regression, open('regmodel.pkl', 'wb'))

pickled_model = pickle.load(open('regmodel.pkl', 'rb'))

# Predict on a new data point (use X_test for example)
sample = X_test[0].reshape(1, -1)
prediction = pickled_model.predict(sample)
print("Prediction for first test sample:", prediction)
