import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# Load the data
file_path = r"C:\Users\GAYATRI\OneDrive\Documents\sem 5\ml\dataset.xlsx"
data = pd.read_excel(file_path)

# Convert text columns to numerical data (Bag of Words)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['Wow!'].tolist() + data['वाह!'].tolist()).toarray()

# Create labels (y) corresponding to each class
y = [0] * len(data) + [1] * len(data)

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Output the shapes of the resulting datasets
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train length:", len(y_train))
print("y_test length:", len(y_test))
