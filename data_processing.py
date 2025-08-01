# data_processing.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

def preprocess_data(df, is_train=True):
    df = df.copy()

    # Fill missing values
    for col in df.select_dtypes(include='number').columns:
        df[col].fillna(df[col].median(), inplace=True)

    for col in df.select_dtypes(include='object').columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # Separate target if training
    if is_train and 'HeartDisease' in df.columns:
        target = df['HeartDisease']
        df = df.drop('HeartDisease', axis=1)
    else:
        target = None

    # One-hot encode categoricals
    df = pd.get_dummies(df, drop_first=True)

    if is_train:
        # 🚨 Save columns for alignment during prediction
        os.makedirs("model", exist_ok=True)
        joblib.dump(df.columns.tolist(), "model/columns.pkl")

        # Fit & save scaler
        scaler = StandardScaler()
        df[df.columns] = scaler.fit_transform(df[df.columns])
        joblib.dump(scaler, "model/scaler.pkl")
    else:
        # 🔁 Load saved columns and align
        saved_columns = joblib.load("model/columns.pkl")
        for col in saved_columns:
            if col not in df.columns:
                df[col] = 0  # add missing columns
        df = df[saved_columns]  # reorder to match training

        # Load and apply scaler
        scaler = joblib.load("model/scaler.pkl")
        df[df.columns] = scaler.transform(df[df.columns])

    # Add back target column in training
    if is_train and target is not None:
        df['HeartDisease'] = target.reset_index(drop=True)

    return df
