import pandas as pd
import numpy as np
import joblib
import argparse
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import os

def load_or_generate_data(data_path):
    if os.path.exists(data_path):
        print(f"[INFO] Loading dataset from {data_path}")
        df = pd.read_csv(data_path)
    else:
        print("[INFO] No dataset found — generating synthetic data...")
        np.random.seed(42)
        n = 1000
        df = pd.DataFrame({
            "age": np.random.randint(18, 70, n),
            "income": np.random.randint(20000, 150000, n),
            "loan_amount": np.random.randint(1000, 50000, n),
            "credit_history": np.random.choice([0, 1], n),
            "dependents": np.random.randint(0, 4, n),
            "credit_score": np.random.choice(["Poor", "Standard", "Good"], n)
        })
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        df.to_csv(data_path, index=False)
        print(f"[INFO] Synthetic data saved to {data_path}")
    return df

def preprocess_data(df):
    # Feature engineering example
    df["debt_to_income"] = df["loan_amount"] / df["income"]
    X = df[["age", "income", "loan_amount", "credit_history", "dependents", "debt_to_income"]]
    y = df["credit_score"]
    return X, y

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(random_state=42))
    ])

    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [5, 10, None]
    }

    print("[INFO] Running GridSearchCV...")
    grid = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[INFO] Best parameters: {grid.best_params_}")
    print(f"[INFO] Test accuracy: {acc:.3f}")
    print("[INFO] Classification Report:\n", classification_report(y_test, y_pred))

    return best_model, grid.best_params_, acc

def save_model(model, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(model, out_path)
    print(f"[INFO] Model saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Credit Score Classification model")
    parser.add_argument("--data", default="data/credit_data.csv", help="Path to input CSV file")
    parser.add_argument("--out", default="artifacts/model.pkl", help="Path to save trained model")
    args = parser.parse_args()

    df = load_or_generate_data(args.data)
    X, y = preprocess_data(df)
    model, params, acc = train_model(X, y)
    save_model(model, args.out)