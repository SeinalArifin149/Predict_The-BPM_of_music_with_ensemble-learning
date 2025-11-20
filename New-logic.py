import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_absolute_error
import optuna
import joblib
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

# ================================
# 1. Load Data
# ================================
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

X = train.drop(["id", "BeatsPerMinute"], axis=1)
y = train["BeatsPerMinute"]
X_test = test.drop(["id"], axis=1)

# ================================
# 2. Optuna Tuning for LGBM
# ================================
def objective_lgbm(trial):
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 20, 100),  # Reduced from 300
        "max_depth": trial.suggest_int("max_depth", 5, 15),      # Modified range
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),  # Increased range
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),     # Reduced range
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),  # Added parameter
        "subsample": trial.suggest_float("subsample", 0.7, 0.9),    # Narrowed range
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 0.9),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 3.0),    # Modified range
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 3.0),  # Modified range
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.5)  # Added parameter
    }
    
    model = LGBMRegressor(
        **params,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        force_col_wise=True  # Added to prevent the warning
    )
    
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    score = -cross_val_score(model, X, y, cv=kf, scoring="neg_mean_absolute_error").mean()
    return score

print("Tuning LightGBM...")
study_lgbm = optuna.create_study(direction="minimize")
study_lgbm.optimize(objective_lgbm, n_trials=30)
best_lgbm_params = study_lgbm.best_params
print("Best LGBM params:", best_lgbm_params)

# ================================
# 3. Optuna Tuning for XGB
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

print("Tuning XGBoost...")
study_xgb = optuna.create_study(direction="minimize")
study_xgb.optimize(objective_xgb, n_trials=30)
best_xgb_params = study_xgb.best_params
print("Best XGB params:", best_xgb_params)

# ================================
# 4. Final Stacking Model
# ================================
final_lgbm = LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.01,
    num_leaves=31,          # Reduced
    max_depth=7,            # Added explicit depth
    min_child_samples=20,   # Added minimum samples
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    min_split_gain=0.1,     # Added minimum gain threshold
    random_state=42,
    n_jobs=-1,
    verbose=-1,
    force_col_wise=True     # Added to prevent the warning
)
final_xgb = XGBRegressor(**best_xgb_params, random_state=42, n_jobs=-1, tree_method="hist")

stack_model = StackingRegressor(
    estimators=[
        ("lgbm", final_lgbm),
        ("xgb", final_xgb),
    ],
    final_estimator=Ridge(alpha=1.0),
    n_jobs=-1
)

# ================================
# 5. Evaluate with Cross-Validation
# ================================
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_score = -cross_val_score(stack_model, X, y, cv=kf, scoring="neg_mean_absolute_error").mean()
print(f"CV MAE: {cv_score:.5f}")

# ================================
# 6. Train Final Model & Predict
# ================================
stack_model.fit(X, y)
preds = stack_model.predict(X_test)

# ================================
# 7. Create Pkl with joblib
# ================================
joblib.dump(stack_model, 'my_model.pkl')

# Save submission
submission = pd.DataFrame({
    "id": test["id"],
    "BeatsPerMinute": preds
})
submission.to_csv("submission_new-logic.csv", index=False)
print("Submission file saved!")
