import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

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

# Initialize lists to store results
k_values = range(1, 12)
accuracies = []

# Train and evaluate kNN classifiers for k from 1 to 11
for k in k_values:
    # Initialize the kNN classifier with current k
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # Fit the model on the training data
    knn.fit(X_train, y_train)
    
    # Predict the labels on the test set
    y_pred = knn.predict(X_test)
    
    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

    # Print accuracy for current k
    print(f"Accuracy for k={k}: {accuracy:.4f}")

# Plot accuracy vs. k
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker='o')
plt.title('Accuracy vs. Number of Neighbors (k)')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.xticks(k_values)
plt.grid(True)
plt.show()
