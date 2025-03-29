# Importing necessary modules
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestRegressor  # Importing Random Forest
from sklearn.metrics import mean_squared_error, mean_absolute_error 

# Importing data
df = pd.read_csv('diabetes.csv') 

# Creating feature variables
X = df.drop('y', axis=1) 
y = df['y'] 

# Creating train and test sets (same split as before)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101) 

# Creating a Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=101)  # 100 trees in the forest

# Training the model
rf_model.fit(X_train, y_train)

# Making predictions
rf_predictions = rf_model.predict(X_test)

# Model evaluation
print('Random Forest Regression Results:')
print('Mean Squared Error:', mean_squared_error(y_test, rf_predictions)) 
print('Mean Absolute Error:', mean_absolute_error(y_test, rf_predictions)) 

# Residual Plot (to check error distribution)
rf_residuals = y_test - rf_predictions
sns.histplot(rf_residuals, kde=True, bins=30)
plt.title("Random Forest Residual Distribution")
plt.show()
