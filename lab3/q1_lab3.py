import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# Load the data
file_path = r"C:\Users\GAYATRI\OneDrive\Documents\sem 5\ml\dataset.xlsx"
data = pd.read_excel(file_path)

# Combine text columns
texts = data['Wow!'].tolist() + data['वाह!'].tolist()

# Convert text to numerical data (Bag of Words)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts).toarray()

# Split data into two classes
split_point = len(data)
class1, class2 = X[:split_point], X[split_point:]

# Calculate centroids and spreads
centroid1, centroid2 = class1.mean(axis=0), class2.mean(axis=0)
spread_class1, spread_class2 = class1.std(axis=0), class2.std(axis=0)

# Calculate the distance between centroids
distance = np.linalg.norm(centroid1 - centroid2)

# Output the results
print("Spread for Class 1:", spread_class1)
print("Spread for Class 2:", spread_class2)
print("Distance between Centroids:", distance)
