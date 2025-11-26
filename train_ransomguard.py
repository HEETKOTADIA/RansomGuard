import os
import json
import time
import joblib
import logging
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import lightgbm as lgb
import xgboost as xgb

# -------------------------------------------------------
# Logging
# -------------------------------------------------------
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler("training.log"),
            logging.StreamHandler()
        ]
    )
setup_logging()

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# -------------------------------------------------------
# Load dataset
# -------------------------------------------------------
DATA_PATH = "final(2).csv"
df = pd.read_csv(DATA_PATH)
logging.info(f"Dataset loaded: {df.shape}")

# -------------------------------------------------------
# Drop useless columns
# -------------------------------------------------------
DROP_COLS = ["SeddAddress", "ExpAddress", "IPaddress", "Threats", "Prediction"]
df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
logging.info(f"Columns after drop: {df.columns.tolist()}")

# -------------------------------------------------------
# Encode label column (Family)
# -------------------------------------------------------
label_encoder = LabelEncoder()
df["Family"] = label_encoder.fit_transform(df["Family"])
joblib.dump(label_encoder, os.path.join(CHECKPOINT_DIR, "label_encoder.joblib"))
logging.info("Saved label encoder for 'Family'.")

# -------------------------------------------------------
# Separate X / y
# -------------------------------------------------------
y = df["Family"]
X = df.drop(columns=["Family"])

# -------------------------------------------------------
# Encode all object columns
# -------------------------------------------------------
cat_encoders = {}
for col in X.columns:
    if X[col].dtype == "object":
        enc = LabelEncoder()
        X[col] = enc.fit_transform(X[col])
        cat_encoders[col] = enc
        joblib.dump(enc, os.path.join(CHECKPOINT_DIR, f"encoder_{col}.joblib"))
        logging.info(f"Encoded categorical column: {col}")

# -------------------------------------------------------
# Preprocessing (numeric)
# -------------------------------------------------------
imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()

X_imp = imputer.fit_transform(X)
X_scaled = scaler.fit_transform(X_imp)

joblib.dump(imputer, os.path.join(CHECKPOINT_DIR, "imputer.joblib"))
joblib.dump(scaler, os.path.join(CHECKPOINT_DIR, "scaler.joblib"))
logging.info("Saved imputer + scaler")

# -------------------------------------------------------
# Split dataset
# -------------------------------------------------------
X_temp, X_test, y_temp, y_test = train_test_split(
    X_scaled, y, test_size=0.10, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.10, random_state=42, stratify=y_temp
)

logging.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# -------------------------------------------------------
# Train LightGBM (OLD + NEW API COMPATIBLE)
# -------------------------------------------------------
logging.info("Training LightGBM...")

lgb_train = lgb.Dataset(X_train, y_train)
lgb_val = lgb.Dataset(X_val, y_val)

lgb_params = {
    "objective": "multiclass",
    "num_class": len(np.unique(y)),
    "metric": "multi_logloss",
    "learning_rate": 0.05,
    "num_leaves": 64,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5
}

lgb_model = lgb.train(
    params=lgb_params,
    train_set=lgb_train,
    valid_sets=[lgb_val],
    num_boost_round=1200,
    callbacks=[
        lgb.early_stopping(80),          # FIXED — works on all versions
        lgb.log_evaluation(100)          # Print every 100 rounds
    ]
)

lgb_model.save_model(os.path.join(CHECKPOINT_DIR, "lgb_model.txt"))
logging.info("Saved LightGBM model")

# -------------------------------------------------------
# Train XGBoost
# -------------------------------------------------------
logging.info("Training XGBoost...")

xg_train = xgb.DMatrix(X_train, y_train)
xg_val = xgb.DMatrix(X_val, y_val)
xg_test = xgb.DMatrix(X_test, y_test)

xgb_params = {
    "objective": "multi:softprob",
    "num_class": len(np.unique(y)),
    "eval_metric": "mlogloss",
    "eta": 0.05,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8
}

xgb_model = xgb.train(
    params=xgb_params,
    dtrain=xg_train,
    evals=[(xg_val, "validation")],
    num_boost_round=900,
    early_stopping_rounds=50,
    verbose_eval=100
)

xgb_model.save_model(os.path.join(CHECKPOINT_DIR, "xgb_model.json"))
logging.info("Saved XGBoost model")

# -------------------------------------------------------
# Evaluate LightGBM (your best model)
# -------------------------------------------------------
logging.info("Evaluating model...")

pred = np.argmax(lgb_model.predict(X_test), axis=1)

acc = accuracy_score(y_test, pred)
report = classification_report(y_test, pred)
cm = confusion_matrix(y_test, pred)

logging.info(f"Accuracy: {acc:.4f}")
logging.info(f"Report:\n{report}")

result = {
    "accuracy": acc,
    "confusion_matrix": cm.tolist(),
    "classification_report": report
}

with open(os.path.join(CHECKPOINT_DIR, "final_report.json"), "w") as f:
    json.dump(result, f, indent=4)

logging.info("Training complete.")
