import kagglehub
path = kagglehub.dataset_download("nanditapore/healthcare-diabetes")
print("Path to dataset files:", path)


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# Load the dataset
url = 'Classification\\Healthcare-Diabetes.csv'
df = pd.read_csv(url)


# Display the first few rows of the dataframe
print(df.head())




# Data preprocessing


# Handle missing values
df = df.dropna()
df = pd.get_dummies(df, columns=['Gender'], drop_first=True)


# Define features and target variable
X = df.drop('Outcome', axis=1)  # Replace 'Outcome' with the actual target column name
y = df['Outcome']  # Replace 'Outcome' with the actual target column name


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Initialize the models
lda = LinearDiscriminantAnalysis()
qda = QuadraticDiscriminantAnalysis()
log_reg = LogisticRegression(max_iter=1000)


# Train and evaluate LDA
lda.fit(X_train, y_train)
y_pred_lda = lda.predict(X_test)
print("LDA Accuracy:", accuracy_score(y_test, y_pred_lda))
print("LDA Classification Report:\n", classification_report(y_test, y_pred_lda))


# Train and evaluate QDA
qda.fit(X_train, y_train)
y_pred_qda = qda.predict(X_test)
print("QDA Accuracy:", accuracy_score(y_test, y_pred_qda))
print("QDA Classification Report:\n", classification_report(y_test, y_pred_qda))


# Train and evaluate Logistic Regression
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log_reg))
print("Logistic Regression Classification Report:\n", classification_report(y_test, y_pred_log_reg))
