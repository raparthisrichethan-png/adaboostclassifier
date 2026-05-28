import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load Dataset
df = pd.read_csv("data/raw/data.csv")

X = df.drop("target", axis=1)
y = df["target"]

# Scaling
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# Base Estimator
base_model = DecisionTreeClassifier(max_depth=1)

# Model
model = AdaBoostClassifier(
    estimator=base_model,
    n_estimators=200,
    learning_rate=0.8,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Save
joblib.dump(model, "models/adaboost_classifier.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("New Model Saved Successfully")