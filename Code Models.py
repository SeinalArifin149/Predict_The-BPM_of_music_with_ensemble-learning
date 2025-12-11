import pandas as pd
import numpy as np
import optuna
import joblib
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# ================================
# 1. Load Data
# ================================
print("Loading data...")
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Pisahkan fitur dan target
X = train.drop(["id", "BeatsPerMinute"], axis=1)
y = train["BeatsPerMinute"]
X_test = test.drop(["id"], axis=1)

# ================================
# 2. Optuna Tuning: LightGBM
# ================================
def objective_lgbm(trial):
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 20, 100),
        "max_depth": trial.suggest_int("max_depth", 5, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
        "subsample": trial.suggest_float("subsample", 0.7, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 0.9),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 3.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 3.0),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.5)
    }
    
    model = LGBMRegressor(
        **params,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        force_col_wise=True
    )
    
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    # Kita cari error terkecil (negative MAE biar dimaximize sama Optuna)
    score = -cross_val_score(model, X, y, cv=kf, scoring="neg_mean_absolute_error").mean()
    return score

print("\n--- 1/3 Tuning LightGBM ---")
study_lgbm = optuna.create_study(direction="minimize")
study_lgbm.optimize(objective_lgbm, n_trials=30)
best_lgbm_params = study_lgbm.best_params
print("Best LGBM params:", best_lgbm_params)

# ================================
# 3. Optuna Tuning: XGBoost
# ================================
def objective_xgb(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05),
        "n_estimators": trial.suggest_int("n_estimators", 500, 3000),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0)
    }
    model = XGBRegressor(**params, random_state=42, n_jobs=-1, tree_method="hist")
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    score = -cross_val_score(model, X, y, cv=kf, scoring="neg_mean_absolute_error").mean()
    return score

print("\n--- 2/3 Tuning XGBoost ---")
study_xgb = optuna.create_study(direction="minimize")
study_xgb.optimize(objective_xgb, n_trials=30)
best_xgb_params = study_xgb.best_params
print("Best XGB params:", best_xgb_params)

# ================================
# 4. Optuna Tuning: CatBoost
# ================================
def objective_cat(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 500, 2000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "random_strength": trial.suggest_float("random_strength", 1e-9, 10),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
        "border_count": trial.suggest_int("border_count", 1, 255),
        "loss_function": "MAE",
        "verbose": 0 
    }
    
    # Task type CPU biar aman di semua laptop, ganti 'GPU' kalau ada NVIDIA
    model = CatBoostRegressor(**params, random_state=42, task_type="CPU") 
    
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    score = -cross_val_score(model, X, y, cv=kf, scoring="neg_mean_absolute_error").mean()
    return score

print("\n--- 3/3 Tuning CatBoost ---")
study_cat = optuna.create_study(direction="minimize")
study_cat.optimize(objective_cat, n_trials=20) 
best_cat_params = study_cat.best_params
print("Best CatBoost params:", best_cat_params)

# ================================
# 5. Final Stacking Model
# ================================
print("\n--- Building Final Stack ---")

# Definisikan ulang model dengan parameter terbaik hasil tuning tadi
final_lgbm = LGBMRegressor(
    n_estimators=3000,        # Jumlah pohon diperbanyak
    learning_rate=0.01,       # Rate dikecilkan biar belajarnya detail
    num_leaves=31,            # Standar emas LightGBM
    max_depth=8,              # Dibatasi biar gak menghafal data (overfit)
    min_child_samples=20,     # Minimal data per daun
    subsample=0.8,            # Ambil 80% data secara acak tiap iterasi
    colsample_bytree=0.8,     # Ambil 80% fitur secara acak
    reg_alpha=0.5,            # L1 Regularization (cegah noise)
    reg_lambda=0.5,           # L2 Regularization
    random_state=42,
    n_jobs=-1,
    verbose=-1,
    force_col_wise=True
)

# 2. XGBOOST: The Classic Powerhouse
final_xgb = XGBRegressor(
    n_estimators=3000,
    learning_rate=0.01,
    max_depth=6,              # Depth 6 adalah sweet spot buat XGBoost
    subsample=0.7,            # Sedikit lebih agresif mengacak data
    colsample_bytree=0.7,
    min_child_weight=1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    tree_method="hist"        # Biar ngebut
)

# 3. CATBOOST: The Finisher
final_cat = CatBoostRegressor(
    iterations=3000,
    learning_rate=0.015,      # Sedikit lebih tinggi dari yang lain
    depth=6,                  # CatBoost sangat optimal di depth 6
    l2_leaf_reg=3,            # Regularisasi standar CatBoost
    bagging_temperature=1,    # Mengurangi varians
    border_count=128,
    loss_function='MAE',      # Fokus ke Mean Absolute Error
    random_state=42,
    verbose=0,
    task_type="CPU"
)

# INI BAGIAN STACKING-NYA
stack_model = StackingRegressor(
    estimators=[
        ("lgbm", final_lgbm),
        ("xgb", final_xgb),
        ("cat", final_cat)
    ],
    final_estimator=Ridge(alpha=1.0), # Meta-learner yang menggabungkan prediksi
    n_jobs=-1,
    cv=5 
)

# ================================
# 6. Evaluate
# ================================
print("Evaluating Stacking Model (CV)...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_score = -cross_val_score(stack_model, X, y, cv=kf, scoring="neg_mean_absolute_error").mean()
print(f"Final Stack CV MAE: {cv_score:.5f}")

# ================================
# 7. Train & Predict
# ================================
print("Training final model on full data...")
stack_model.fit(X, y)
preds = stack_model.predict(X_test)

# ================================
# 8. Save Result
# ================================
joblib.dump(stack_model, 'my_super_stack_model.pkl')

submission = pd.DataFrame({
    "id": test["id"],
    "BeatsPerMinute": preds
})
submission.to_csv("submission_stacking_lgbm_xgb_cat.csv", index=False)
print("Selesai! File 'submission_stacking_lgbm_xgb_cat.csv' siap di-submit.")