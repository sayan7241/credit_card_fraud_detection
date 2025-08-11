import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
credit_card_data = pd.read_csv(r'C:\Users\RITH\Desktop\PROJECTS\credit card\creditcard (2).csv')  # Change path if needed
print(credit_card_data.head())

# Summary statistics
print(credit_card_data.describe().transpose())

# Dataset information
print(credit_card_data.info())

# Check missing values
print(credit_card_data.isnull().sum())

# Plot histograms
credit_card_data.hist(figsize=(20, 20))
plt.show()

# Standardize the variables
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(credit_card_data.drop(["Class"], axis=1)))
y = credit_card_data["Class"]

# Train-Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Using KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)

# Predictions and Evaluations
from sklearn.metrics import classification_report, confusion_matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, pred))
print("\nClassification Report:")
print(classification_report(y_test, pred))
