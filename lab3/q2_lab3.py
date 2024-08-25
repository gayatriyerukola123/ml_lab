import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

# Load the data
file_path = r"C:\Users\GAYATRI\OneDrive\Documents\sem 5\ml\dataset.xlsx"
data = pd.read_excel(file_path)

# Select a feature from the dataset (let's assume it's the first feature from the Bag of Words)
# Convert text columns to numerical data (Bag of Words)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['Wow!'].tolist() + data['वाह!'].tolist()).toarray()

# Assuming we're analyzing the first feature (first word/column)
feature = X[:, 0]

# Plot histogram for the selected feature
plt.hist(feature, bins=10, color='blue', edgecolor='black')
plt.title("Histogram of Feature 1")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

# Calculate mean and variance of the selected feature
mean = np.mean(feature)
variance = np.var(feature)

# Output the results
print("Mean of Feature 1:", mean)
print("Variance of Feature 1:", variance)
