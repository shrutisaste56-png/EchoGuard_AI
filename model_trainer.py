
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("dataset.csv")

X = df.drop("label", axis=1)
y = df["label"]

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)
print("Real samples:", (y == 1).sum())
print("Fake samples:", (y == 0).sum())

print("\nTotal samples:", len(df))
print("\nLabel distribution:")
print(y.value_counts())

# Check dataset
if len(df) < 4 or y.nunique() < 2:
    raise ValueError("Need more audio samples from both classes.")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42, stratify=y
)

# Train Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy * 100, "%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "voice_model.pkl")

print("\nModel saved successfully as voice_model.pkl")
