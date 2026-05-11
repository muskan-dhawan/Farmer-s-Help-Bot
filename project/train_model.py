import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
import xgboost as xgb
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

# ===============================
# 🟢 STEP 1: LOAD DATA
# ===============================
df = pd.read_csv(r'c:\Users\rahul\Desktop\ethic\data\Crop_recommendation.csv')

features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
X = df[features]
y_raw = df['label']

# ===============================
# 🟢 STEP 2: ENCODE LABEL
# ===============================
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

# ===============================
# 🟢 STEP 3: TRAIN-TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===============================
# 🟢 STEP 4: BASE MODEL PIPELINES
# ===============================
rf_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1))
])

xgb_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('xgb', xgb.XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1
    ))
])

et_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('et', ExtraTreesClassifier(n_estimators=300, random_state=42, n_jobs=-1))
])

lgb_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('lgb', lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        random_state=42
    ))
])

svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', probability=True, random_state=42))
])

# ===============================
# 🟢 STEP 5: PREPARE FINAL MODEL
# ===============================
# User requested Random Forest as the best model instead of Stacking
final_model = rf_pipeline

# ===============================
# 🟢 STEP 6: TRAIN
# ===============================
final_model.fit(X_train, y_train)

# ===============================
# 🟢 STEP 7: PREDICTIONS
# ===============================
y_pred = final_model.predict(X_test)

# ===============================
# 🟢 STEP 8: METRICS
# ===============================
accuracy = accuracy_score(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n===== PERFORMANCE =====")
print(f"Accuracy : {accuracy:.4f}")

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_
))

print("\n===== EXTRA METRICS (for analysis only) =====")
print(f"MSE  : {mse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# ===============================
# 🟢 STEP 9: TOP-3 ACCURACY
# ===============================
def top3_accuracy(model, X, y):
    probs = model.predict_proba(X)
    top3 = np.argsort(probs, axis=1)[:, -3:]
    
    correct = 0
    for i in range(len(y)):
        if y[i] in top3[i]:
            correct += 1
    
    return correct / len(y)

top3_acc = top3_accuracy(final_model, X_test, y_test)
print(f"\nTop-3 Accuracy: {top3_acc:.4f}")

# ===============================
# 🟢 STEP 10: TOP-3 PREDICTION FUNCTION
# ===============================
def predict_top3(input_data):
    probs = final_model.predict_proba([input_data])[0]
    top3_idx = np.argsort(probs)[-3:][::-1]

    results = []
    for idx in top3_idx:
        crop = label_encoder.inverse_transform([idx])[0]
        confidence = probs[idx]
        results.append((crop, round(confidence * 100, 2)))

    return results

# Example
print("\nExample Prediction:")
print(predict_top3(X_test.iloc[0]))

# ===============================
# 🟢 STEP 11: CROSS VALIDATION
# ===============================
cv_score = cross_val_score(final_model, X_train, y_train, cv=5).mean()
print(f"\nCross Validation Score: {cv_score:.4f}")

# ===============================
# 🟢 STEP 12: SAVE PIPELINE
# ===============================
os.makedirs('models', exist_ok=True)

pickle.dump(final_model, open('models/crop_pipeline.pkl', 'wb'))
pickle.dump(label_encoder, open('models/label_encoder.pkl', 'wb'))
pickle.dump(features, open('models/features.pkl', 'wb'))

print("\nFINAL PIPELINE (RANDOM FOREST) SAVED SUCCESSFULLY")