# importing modules and packages 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, mean_absolute_error 
from sklearn import preprocessing 

# importing data 
df = pd.read_csv('diabetes.csv') 

# creating feature variables 
X = df.drop('y',axis= 1) 
y = df['y'] 

# creating train and test sets 
X_train, X_test, y_train, y_test = train_test_split( 
	X, y, test_size=0.3, random_state=101) 

# creating a regression model 
model = LinearRegression() 

model.fit(X_train,y_train)

predictions = model.predict(X_test) 
  
# model evaluation 
print('mean_squared_error : ', mean_squared_error(y_test, predictions)) 
print('mean_absolute_error : ', mean_absolute_error(y_test, predictions)) 

residuals = y_test - predictions
sns.histplot(residuals, kde=True)
plt.title("Residual Distribution")
plt.show()