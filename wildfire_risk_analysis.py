"""
Wildfire Risk Analytics and Prediction
Internship Project

IMPORTANT:
The included CSV is SYNTHETIC data created for demonstration/academic use.
It is not an official wildfire dataset and should not be presented as real-world observations.

Run:
    pip install -r requirements.txt
    python wildfire_risk_analysis.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

DATA_FILE = Path("wildfire_risk_data.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

FEATURES = [
    "temperature_c", "humidity_pct", "wind_speed_kmh", "rainfall_mm",
    "vegetation_dryness", "population_density", "distance_to_road_km",
    "elevation_m"
]

def load_data():
    df = pd.read_csv(DATA_FILE)
    print("Dataset shape:", df.shape)
    print("\nMissing values:\n", df.isna().sum())
    return df

def exploratory_analysis(df):
    print("\nDescriptive statistics:")
    print(df[FEATURES + ["wildfire_risk"]].describe())

    print("\nRisk by region:")
    print(df.groupby("region")["wildfire_risk"].mean().sort_values(ascending=False))

    # Correlation heatmap
    corr = df[FEATURES + ["wildfire_risk"]].corr()
    plt.figure(figsize=(10, 7))
    plt.imshow(corr, aspect="auto")
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=75, ha="right")
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_matrix.png", dpi=160)
    plt.close()

    # Risk distribution
    counts = df["wildfire_risk"].value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    plt.bar(["Low / No Fire", "High / Fire"], counts.values)
    plt.ylabel("Number of Areas")
    plt.title("Synthetic Wildfire Risk Distribution")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "risk_distribution.png", dpi=160)
    plt.close()

def train_predictive_model(df):
    X = df[FEATURES]
    y = df["wildfire_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=250,
        random_state=42,
        class_weight="balanced",
        max_depth=10
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print("\nModel accuracy:", round(accuracy_score(y_test, pred), 4))
    print("\nClassification report:\n", classification_report(y_test, pred))

    cm = confusion_matrix(y_test, pred)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.colorbar()
    plt.xticks([0, 1], ["Predicted Low", "Predicted High"])
    plt.yticks([0, 1], ["Actual Low", "Actual High"])
    plt.xlabel("Prediction")
    plt.ylabel("Actual")
    plt.title("Random Forest Confusion Matrix")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    importance = pd.Series(
        model.feature_importances_, index=FEATURES
    ).sort_values(ascending=False)

    plt.figure(figsize=(8, 5))
    importance.sort_values().plot(kind="barh")
    plt.xlabel("Importance")
    plt.title("Random Forest Feature Importance")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=160)
    plt.close()

    return model

def clustering_analysis(df):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURES])

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df["risk_cluster"] = kmeans.fit_predict(X_scaled)

    summary = df.groupby("risk_cluster")[FEATURES + ["wildfire_risk"]].mean()
    print("\nCluster summary:\n", summary)
    summary.to_csv(OUTPUT_DIR / "cluster_summary.csv")

    return df

def main():
    df = load_data()
    exploratory_analysis(df)
    model = train_predictive_model(df)
    df = clustering_analysis(df)

    df.to_csv(OUTPUT_DIR / "wildfire_risk_with_clusters.csv", index=False)
    print("\nAnalysis complete. Check the outputs/ folder.")

if __name__ == "__main__":
    main()
