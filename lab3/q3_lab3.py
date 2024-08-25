import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.feature_extraction.text import CountVectorizer

# Load the data
file_path = r"C:\Users\GAYATRI\OneDrive\Documents\sem 5\ml\dataset.xlsx"
data = pd.read_excel(file_path)

# Convert text columns to numerical data (Bag of Words)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['Wow!'].tolist() + data['वाह!'].tolist()).toarray()

# Select any two feature vectors (rows) from the dataset
feature_vector1 = X[0]  # First row
feature_vector2 = X[1]  # Second row

# Calculate Minkowski distances for r values from 1 to 10
r_values = range(1, 11)
minkowski_distances = [distance.minkowski(feature_vector1, feature_vector2, p=r) for r in r_values]

# Plot the Minkowski distances against r values
plt.plot(r_values, minkowski_distances, marker='o')
plt.title("Minkowski Distance between Two Feature Vectors")
plt.xlabel("r (Order of the Minkowski Distance)")
plt.ylabel("Minkowski Distance")
plt.grid(True)
plt.show()

# Output the distances for reference
for r, dist in zip(r_values, minkowski_distances):
    print(f"Minkowski Distance with r={r}: {dist}")
