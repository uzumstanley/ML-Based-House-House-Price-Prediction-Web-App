import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # MATLAB-like way of plotting
from scipy import stats
import pickle
# sklearn package for machine learning in python:
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv('houseprice_data.csv')
print(df.head(20))

X = df.iloc[:, [1,3,8,12,13]].values
y = df.iloc[:, 0].values

df.drop('sqft_living15', axis=1, inplace=True)
df.drop('sqft_above', axis=1, inplace=True)

# Identify outliers using Z-score
z_scores = np.abs(stats.zscore(df[['sqft_living', 'price']]))
outliers = (z_scores > 2)

# Remove outliers
df_no_outliers = df[~outliers.any(axis=1)]

print("Original DataFrame shape:", df.shape)
print("New DataFrame shape:", df_no_outliers.shape) 

# split the data into training and test sets:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 1/3,
random_state=0)

# fit the linear least-squres regression line to the training data:
regr = LinearRegression()
regr.fit(X_train, y_train)
# The coefficients
print('Coefficients: ', regr.coef_)
# The intercept
print('Intercept: ', regr.intercept_)
# The mean squared error
print('Mean squared error: %.8f'
% mean_squared_error(y_test, regr.predict(X_test)))
# The R^2 value:
print('Coefficient of determination: %.2f'
% r2_score(y_test, regr.predict(X_test)))


# Save the trained model to a file using Pickle
with open('Multiplelinear_regression_model.pkl', 'wb') as file:  # 'wb' means write in binary mode
    pickle.dump(regr, file)


print("Model has been saved to 'Muliplelinear_regression_model.pkl'.")

