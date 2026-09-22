# ============================================================
# CUSTOMER SEGMENTATION ANALYSIS
# THIRANEX PROJECT
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# 1. CREATE CUSTOMER DATASET
# ============================================================

print("=" * 60)
print("CUSTOMER SEGMENTATION ANALYSIS")
print("=" * 60)

np.random.seed(42)

n = 500

data = {
    "CustomerID": range(1001, 1001 + n),
    "Age": np.random.randint(18, 65, n),
    "Annual_Income": np.random.randint(20000, 120000, n),
    "Spending_Score": np.random.randint(1, 101, n),
    "Purchase_Frequency": np.random.randint(1, 31, n),
    "Total_Purchases": np.random.randint(1, 101, n),
    "Online_Purchases": np.random.randint(0, 61, n),
    "Website_Visits": np.random.randint(1, 51, n)
}

df = pd.DataFrame(data)

df.to_csv("customers.csv", index=False)

print("\nDataset created successfully!")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


# ============================================================
# 2. DATA UNDERSTANDING
# ============================================================

print("\n" + "=" * 60)
print("DATA UNDERSTANDING")
print("=" * 60)

print("\nFirst 5 Customers:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nStatistical Summary:")
print(df.describe())


# ============================================================
# 3. DATA CLEANING
# ============================================================

print("\n" + "=" * 60)
print("DATA CLEANING")
print("=" * 60)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

df = df.drop_duplicates()

print("\nData cleaning completed.")


# ============================================================
# 4. EXPLORATORY DATA ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)


# Age distribution
plt.figure(figsize=(8, 5))

sns.histplot(
    df["Age"],
    bins=20,
    kde=True
)

plt.title("Customer Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.savefig("age_distribution.png")
plt.close()


# Income distribution
plt.figure(figsize=(8, 5))

sns.histplot(
    df["Annual_Income"],
    bins=20,
    kde=True
)

plt.title("Annual Income Distribution")
plt.xlabel("Annual Income")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.savefig("income_distribution.png")
plt.close()


# Spending distribution
plt.figure(figsize=(8, 5))

sns.histplot(
    df["Spending_Score"],
    bins=20,
    kde=True
)

plt.title("Customer Spending Score")
plt.xlabel("Spending Score")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.savefig("spending_distribution.png")
plt.close()

print("\nEDA completed.")


# ============================================================
# 5. CORRELATION ANALYSIS
# ============================================================

print("\nCreating correlation heatmap...")

correlation = df.drop(
    "CustomerID",
    axis=1
).corr()

plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Customer Feature Correlation")

plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.close()

print("Correlation analysis completed.")


# ============================================================
# 6. SELECT FEATURES FOR CLUSTERING
# ============================================================

features = [
    "Age",
    "Annual_Income",
    "Spending_Score",
    "Purchase_Frequency",
    "Total_Purchases",
    "Online_Purchases",
    "Website_Visits"
]

X = df[features]

print("\nFeatures selected for clustering:")
for feature in features:
    print("-", feature)


# ============================================================
# 7. FEATURE SCALING
# ============================================================

print("\nScaling features...")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("Feature scaling completed.")


# ============================================================
# 8. ELBOW METHOD
# ============================================================

print("\nCalculating Elbow Method...")

inertia = []

for k in range(2, 11):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    inertia.append(model.inertia_)


plt.figure(figsize=(8, 5))

plt.plot(
    range(2, 11),
    inertia,
    marker="o"
)

plt.title("Elbow Method for Optimal Number of Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")

plt.tight_layout()
plt.savefig("elbow_method.png")
plt.close()

print("Elbow Method completed.")


# ============================================================
# 9. K-MEANS CLUSTERING
# ============================================================

print("\nPerforming K-Means clustering...")

optimal_k = 4

kmeans = KMeans(
    n_clusters=optimal_k,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)

print("K-Means clustering completed.")


# ============================================================
# 10. CLUSTER SIZE
# ============================================================

print("\n" + "=" * 60)
print("CUSTOMER DISTRIBUTION")
print("=" * 60)

cluster_counts = df["Cluster"].value_counts().sort_index()

print(cluster_counts)


# ============================================================
# 11. SILHOUETTE SCORE
# ============================================================

silhouette = silhouette_score(
    X_scaled,
    df["Cluster"]
)

print("\nSilhouette Score:", round(silhouette, 3))


# ============================================================
# 12. CLUSTER SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("CLUSTER SUMMARY")
print("=" * 60)

cluster_summary = df.groupby(
    "Cluster"
)[features].mean()

print(cluster_summary.round(2))

cluster_summary.to_csv(
    "cluster_summary.csv"
)


# ============================================================
# 13. CUSTOMER SEGMENT VISUALIZATION
# ============================================================

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="Annual_Income",
    y="Spending_Score",
    hue="Cluster",
    palette="viridis",
    s=80
)

plt.title("Customer Segmentation")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")

plt.tight_layout()
plt.savefig("customer_segments.png")
plt.close()


# ============================================================
# 14. SEGMENT SIZE VISUALIZATION
# ============================================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Cluster"
)

plt.title("Number of Customers in Each Segment")
plt.xlabel("Customer Segment")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.savefig("segment_sizes.png")
plt.close()


# ============================================================
# 15. SPENDING BY SEGMENT
# ============================================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Cluster",
    y="Spending_Score"
)

plt.title("Spending Score by Customer Segment")
plt.xlabel("Customer Segment")
plt.ylabel("Spending Score")

plt.tight_layout()
plt.savefig("spending_by_segment.png")
plt.close()


# ============================================================
# 16. PURCHASE FREQUENCY
# ============================================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Cluster",
    y="Purchase_Frequency"
)

plt.title("Purchase Frequency by Customer Segment")
plt.xlabel("Customer Segment")
plt.ylabel("Purchase Frequency")

plt.tight_layout()
plt.savefig("purchase_frequency.png")
plt.close()


# ============================================================
# 17. BUSINESS INSIGHTS
# ============================================================

print("\n" + "=" * 60)
print("BUSINESS INSIGHTS")
print("=" * 60)

overall_income = cluster_summary[
    "Annual_Income"
].mean()

overall_spending = cluster_summary[
    "Spending_Score"
].mean()

overall_frequency = cluster_summary[
    "Purchase_Frequency"
].mean()


for cluster in cluster_summary.index:

    income = cluster_summary.loc[
        cluster,
        "Annual_Income"
    ]

    spending = cluster_summary.loc[
        cluster,
        "Spending_Score"
    ]

    frequency = cluster_summary.loc[
        cluster,
        "Purchase_Frequency"
    ]

    print("\nCluster", cluster)
    print("-" * 40)

    if (
        income > overall_income
        and spending > overall_spending
    ):

        print("Customer Type: HIGH-VALUE CUSTOMERS")

        print(
            "Strategy: Premium products, "
            "loyalty rewards and exclusive offers."
        )

    elif frequency > overall_frequency:

        print("Customer Type: FREQUENT CUSTOMERS")

        print(
            "Strategy: Loyalty programs and "
            "personalized recommendations."
        )

    elif (
        spending < overall_spending
        and frequency < overall_frequency
    ):

        print("Customer Type: LOW-ENGAGEMENT CUSTOMERS")

        print(
            "Strategy: Discounts, promotional "
            "campaigns and re-engagement offers."
        )

    else:

        print("Customer Type: REGULAR CUSTOMERS")

        print(
            "Strategy: Personalized offers "
            "to increase spending and engagement."
        )


# ============================================================
# 18. SAVE FINAL DATA
# ============================================================

df.to_csv(
    "customer_segments.csv",
    index=False
)


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\nFiles created:")
print("1. customers.csv")
print("2. customer_segments.csv")
print("3. cluster_summary.csv")
print("4. age_distribution.png")
print("5. income_distribution.png")
print("6. spending_distribution.png")
print("7. correlation_heatmap.png")
print("8. elbow_method.png")
print("9. customer_segments.png")
print("10. segment_sizes.png")
print("11. spending_by_segment.png")
print("12. purchase_frequency.png")