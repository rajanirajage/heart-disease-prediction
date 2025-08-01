import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os
from data_processing import preprocess_data

print("Starting training script...")

# Load dataset
df = pd.read_csv("Heart_disease.csv")
print("Loaded dataset with shape:", df.shape)

# Preprocess
df = preprocess_data(df, is_train=True)
print("Data after preprocessing:", df.shape)

# Check if target exists
if 'HeartDisease' not in df.columns:
    print("ERROR: 'HeartDisease' column missing after preprocessing")
    exit()

# Split into X and y
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print("Split data into training and validation sets")

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Model trained successfully")

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/heart_disease_model.pkl")
print("Model saved to 'model/heart_disease_model.pkl'")

# Evaluate
y_pred = model.predict(X_val)
print("\nModel Evaluation Report:\n")
print(classification_report(y_val, y_pred))
