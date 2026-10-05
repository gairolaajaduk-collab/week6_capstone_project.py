# ============================================================
# WEEK 6: INTEGRATIVE CAPSTONE PROJECT
# Titanic Survival Prediction and Passenger Segmentation
#
# Supervised Learning  : Random Forest Classifier
# Unsupervised Learning: K-Means Clustering
# ============================================================


# ------------------------------------------------------------
# 1. IMPORT LIBRARIES
# ------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    silhouette_score
)


# ------------------------------------------------------------
# 2. DATA ACQUISITION
# ------------------------------------------------------------

print("Loading Titanic dataset...")

# Titanic dataset from Seaborn
df = sns.load_dataset("titanic")

print("\nDataset loaded successfully.")

print("\nFirst 5 rows:")
print(df.head())


# ------------------------------------------------------------
# 3. BASIC DATA INFORMATION
# ------------------------------------------------------------

print("\n========================================")
print("DATASET INFORMATION")
print("========================================")

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nDataset information:")
print(df.info())

print("\nStatistical summary:")
print(df.describe())


# ------------------------------------------------------------
# 4. CHECK MISSING VALUES
# ------------------------------------------------------------

print("\n========================================")
print("MISSING VALUES")
print("========================================")

print(df.isnull().sum())


# ------------------------------------------------------------
# 5. DATA CLEANING
# ------------------------------------------------------------

# Create a copy
data = df.copy()

# Fill missing age values using median
data["age"] = data["age"].fillna(
    data["age"].median()
)

# Fill missing embarked values using mode
data["embarked"] = data["embarked"].fillna(
    data["embarked"].mode()[0]
)

# Fill missing embark_town values
data["embark_town"] = data["embark_town"].fillna(
    data["embark_town"].mode()[0]
)

# Drop deck because it contains many missing values
data = data.drop(columns=["deck"])

print("\nMissing values after cleaning:")
print(data.isnull().sum())


# ------------------------------------------------------------
# 6. FEATURE ENGINEERING
# ------------------------------------------------------------

# Create family size
data["family_size"] = (
    data["sibsp"] + data["parch"] + 1
)

# Create a simple travel group category
data["is_alone"] = (
    data["family_size"] == 1
).astype(int)

# Create age group
data["age_group"] = pd.cut(
    data["age"],
    bins=[0, 12, 18, 35, 60, 100],
    labels=[
        "Child",
        "Teenager",
        "Adult",
        "Middle_Aged",
        "Senior"
    ]
)

print("\nNew features created.")

print(
    data[
        [
            "age",
            "sibsp",
            "parch",
            "family_size",
            "is_alone",
            "age_group"
        ]
    ].head()
)


# ------------------------------------------------------------
# 7. EXPLORATORY DATA ANALYSIS
# ------------------------------------------------------------

print("\n========================================")
print("EXPLORATORY DATA ANALYSIS")
print("========================================")

print("\nSurvival counts:")
print(data["survived"].value_counts())

print("\nSurvival rate:")
print(data["survived"].mean())


# ------------------------------------------------------------
# 8. SURVIVAL COUNT VISUALIZATION
# ------------------------------------------------------------

plt.figure(figsize=(7, 5))

sns.countplot(
    data=data,
    x="survived"
)

plt.title("Titanic Survival Count")

plt.xlabel("Survived (0 = No, 1 = Yes)")

plt.ylabel("Number of Passengers")

plt.show()


# ------------------------------------------------------------
# 9. SURVIVAL BY GENDER
# ------------------------------------------------------------

plt.figure(figsize=(7, 5))

sns.countplot(
    data=data,
    x="sex",
    hue="survived"
)

plt.title("Survival by Gender")

plt.xlabel("Gender")

plt.ylabel("Number of Passengers")

plt.show()


# ------------------------------------------------------------
# 10. SURVIVAL BY PASSENGER CLASS
# ------------------------------------------------------------

plt.figure(figsize=(7, 5))

sns.countplot(
    data=data,
    x="class",
    hue="survived"
)

plt.title("Survival by Passenger Class")

plt.xlabel("Passenger Class")

plt.ylabel("Number of Passengers")

plt.show()


# ------------------------------------------------------------
# 11. AGE DISTRIBUTION
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    data=data,
    x="age",
    bins=30,
    kde=True
)

plt.title("Age Distribution of Passengers")

plt.xlabel("Age")

plt.ylabel("Number of Passengers")

plt.show()


# ------------------------------------------------------------
# 12. CORRELATION HEATMAP
# ------------------------------------------------------------

numeric_data = data.select_dtypes(
    include=np.number
)

plt.figure(figsize=(10, 7))

sns.heatmap(
    numeric_data.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.show()


# ============================================================
# PART A: SUPERVISED LEARNING
# ============================================================

print("\n\n========================================")
print("SUPERVISED LEARNING")
print("========================================")


# ------------------------------------------------------------
# 13. SELECT FEATURES
# ------------------------------------------------------------

features = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "family_size",
    "is_alone",
    "embarked"
]

model_data = data[features + ["survived"]].copy()


# Convert categorical variables into numerical variables
model_data = pd.get_dummies(
    model_data,
    columns=["sex", "embarked"],
    drop_first=True
)


print("\nFeatures used for prediction:")
print(model_data.columns)


# ------------------------------------------------------------
# 14. SPLIT FEATURES AND TARGET
# ------------------------------------------------------------

X = model_data.drop(
    columns=["survived"]
)

y = model_data["survived"]


# ------------------------------------------------------------
# 15. TRAIN-TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))

print("Testing samples:", len(X_test))


# ------------------------------------------------------------
# 16. FEATURE SCALING
# ------------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# ------------------------------------------------------------
# 17. CREATE RANDOM FOREST MODEL
# ------------------------------------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    random_state=42
)


# ------------------------------------------------------------
# 18. TRAIN MODEL
# ------------------------------------------------------------

print("\nTraining Random Forest model...")

model.fit(
    X_train_scaled,
    y_train
)

print("Model training completed.")


# ------------------------------------------------------------
# 19. MAKE PREDICTIONS
# ------------------------------------------------------------

y_pred = model.predict(
    X_test_scaled
)


# ------------------------------------------------------------
# 20. MODEL ACCURACY
# ------------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n========================================")
print("SUPERVISED MODEL RESULTS")
print("========================================")

print(
    "Accuracy:",
    round(accuracy * 100, 2),
    "%"
)


# ------------------------------------------------------------
# 21. CLASSIFICATION REPORT
# ------------------------------------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ------------------------------------------------------------
# 22. CONFUSION MATRIX
# ------------------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")

print(cm)


plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Random Forest Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()


# ------------------------------------------------------------
# 23. FEATURE IMPORTANCE
# ------------------------------------------------------------

importance = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)


print("\nFeature Importance:")

print(importance_df)


plt.figure(figsize=(10, 6))

sns.barplot(
    data=importance_df,
    x="Importance",
    y="Feature"
)

plt.title(
    "Random Forest Feature Importance"
)

plt.show()


# ============================================================
# PART B: UNSUPERVISED LEARNING
# ============================================================

print("\n\n========================================")
print("UNSUPERVISED LEARNING")
print("========================================")


# ------------------------------------------------------------
# 24. SELECT FEATURES FOR CLUSTERING
# ------------------------------------------------------------

cluster_features = data[
    [
        "age",
        "fare",
        "family_size",
        "pclass"
    ]
].copy()


# ------------------------------------------------------------
# 25. SCALE CLUSTERING DATA
# ------------------------------------------------------------

cluster_scaler = StandardScaler()

cluster_scaled = cluster_scaler.fit_transform(
    cluster_features
)


# ------------------------------------------------------------
# 26. CREATE K-MEANS MODEL
# ------------------------------------------------------------

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)


# ------------------------------------------------------------
# 27. TRAIN K-MEANS
# ------------------------------------------------------------

clusters = kmeans.fit_predict(
    cluster_scaled
)

print("\nK-Means clustering completed.")


# Add cluster labels to data
data["cluster"] = clusters


# ------------------------------------------------------------
# 28. SILHOUETTE SCORE
# ------------------------------------------------------------

silhouette = silhouette_score(
    cluster_scaled,
    clusters
)

print(
    "\nSilhouette Score:",
    round(silhouette, 3)
)


# ------------------------------------------------------------
# 29. DISPLAY CLUSTER INFORMATION
# ------------------------------------------------------------

print("\nCluster counts:")

print(
    data["cluster"].value_counts()
)


print("\nAverage values for each cluster:")

cluster_summary = data.groupby(
    "cluster"
)[
    [
        "age",
        "fare",
        "family_size",
        "pclass",
        "survived"
    ]
].mean()

print(cluster_summary)


# ------------------------------------------------------------
# 30. VISUALIZE CLUSTERS
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=data,
    x="age",
    y="fare",
    hue="cluster",
    palette="viridis"
)

plt.title(
    "Passenger Segmentation using K-Means"
)

plt.xlabel("Age")

plt.ylabel("Fare")

plt.show()


# ============================================================
# PART C: FINAL INSIGHTS
# ============================================================

print("\n\n========================================")
print("FINAL INSIGHTS")
print("========================================")


# Gender survival rate
gender_survival = data.groupby(
    "sex"
)["survived"].mean()

print("\nSurvival rate by gender:")

print(gender_survival)


# Class survival rate
class_survival = data.groupby(
    "class"
)["survived"].mean()

print("\nSurvival rate by passenger class:")

print(class_survival)


# Average fare
print(
    "\nAverage fare:",
    round(data["fare"].mean(), 2)
)


# Average age
print(
    "Average age:",
    round(data["age"].mean(), 2)
)


# ------------------------------------------------------------
# 31. SAVE CLEANED DATASET
# ------------------------------------------------------------

data.to_csv(
    "titanic_cleaned_data.csv",
    index=False
)

print(
    "\nCleaned dataset saved as:",
    "titanic_cleaned_data.csv"
)


# ------------------------------------------------------------
# 32. SAVE SUPERVISED MODEL
# ------------------------------------------------------------

import joblib

joblib.dump(
    model,
    "titanic_random_forest_model.pkl"
)

print(
    "Random Forest model saved as:",
    "titanic_random_forest_model.pkl"
)


# ------------------------------------------------------------
# 33. FINAL SUMMARY
# ------------------------------------------------------------

print("\n========================================")
print("CAPSTONE PROJECT SUMMARY")
print("========================================")

print(
    "Dataset: Titanic Passenger Dataset"
)

print(
    "Supervised Model: Random Forest Classifier"
)

print(
    "Supervised Accuracy:",
    round(accuracy * 100, 2),
    "%"
)

print(
    "Unsupervised Model: K-Means Clustering"
)

print(
    "Silhouette Score:",
    round(silhouette, 3)
)

print(
    "Total Passengers:",
    len(data)
)

print("========================================")

print("\nCapstone project completed successfully.")
