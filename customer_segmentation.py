# Customer Segmentation using K-Means

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# -------------------------------
# 1. LOAD DATA
# -------------------------------
# Use Mall Customers dataset CSV (columns: CustomerID, Gender, Age, Annual Income (k$), Spending Score (1-100))

df = pd.read_csv("Mall_Customers.csv")

print("First 5 rows:")
print(df.head())

# -------------------------------
# 2. DATA CLEANING
# -------------------------------
# Drop unnecessary column
df = df.drop(columns=["CustomerID"], errors='ignore')

# Convert Gender to numeric
df['Genre'] = df['Genre'].map({'Male': 0, 'Female': 1})
# Handle missing values (if any)
df = df.dropna()

# -------------------------------
# 3. FEATURE SELECTION
# -------------------------------
features = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# -------------------------------
# 4. FEATURE SCALING
# -------------------------------
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# -------------------------------
# 5. ELBOW METHOD
# -------------------------------
wcss = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

plt.figure()
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("WCSS")
plt.show()

# Choose optimal k (typically 5 for this dataset)
k = 5

# -------------------------------
# 6. APPLY K-MEANS
# -------------------------------
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(scaled_features)

df['Cluster'] = clusters

print("\nCluster Assigned Data:")
print(df.head())

# -------------------------------
# 7. VISUALIZATION
# -------------------------------
plt.figure()

for i in range(k):
    plt.scatter(
        df[df['Cluster'] == i]['Annual Income (k$)'],
        df[df['Cluster'] == i]['Spending Score (1-100)'],
        label=f'Cluster {i}'
    )

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score")
plt.title("Customer Segments")
plt.legend()
plt.show()

# -------------------------------
# 8. CLUSTER PROFILING
# -------------------------------
cluster_profile = df.groupby('Cluster').mean()
print("\nCluster Profile:")
print(cluster_profile)

# -------------------------------
# 9. SAVE RESULTS
# -------------------------------
df.to_csv("customer_segments_output.csv", index=False)

# -------------------------------
# 10. GENERATE REPORT
# -------------------------------
print("\n--- CUSTOMER SEGMENT REPORT ---")

for i in range(k):
    segment = df[df['Cluster'] == i]
    
    avg_age = segment['Age'].mean()
    avg_income = segment['Annual Income (k$)'].mean()
    avg_score = segment['Spending Score (1-100)'].mean()
    
    print(f"\nSegment {i}:")
    print(f"Average Age: {avg_age:.1f}")
    print(f"Average Income: {avg_income:.1f}")
    print(f"Spending Score: {avg_score:.1f}")
    
    # Marketing insight
    if avg_score > 70:
        print("Insight: High spenders → Target with premium offers.")
    elif avg_score < 40:
        print("Insight: Low spenders → Use discounts & engagement strategies.")
    else:
        print("Insight: Moderate customers → Loyalty programs recommended.")