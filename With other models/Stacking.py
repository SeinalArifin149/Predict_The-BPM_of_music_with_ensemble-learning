import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, StackingRegressor,VotingRegressor
from sklearn.ensemble import VotingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

# load data set

train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")
sample=pd.read_csv("sample_submission.csv")


# crop data with test 
# if u train all data,u change var in train_pd to train and comment this var

# train=train_pd.sample (frac=0.1, random_state=42)

# stacking mode
# from data train
y = train["BeatsPerMinute"]

x = train.drop(["id", "BeatsPerMinute"], axis=1)
# from data test
x_test_kaggle= test.drop(["id"], axis=1)
test_ids= test["id"]

# split data 
x_train, x_valid, y_train, y_valid = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# create model
 
rf= RandomForestRegressor(
    n_estimators=500,       # banyak pohon, lebih stabil
    max_depth=25,           # batasi kedalaman biar ga overfit
    min_samples_split=10,   # minimal data untuk split
    min_samples_leaf=4,     # minimal data di leaf
    max_features='sqrt',    # ambil sqrt fitur tiap split, biar random
    n_jobs=-1,
    random_state=42
    )
xgb= XGBRegressor(
    n_estimators=1000,       # banyak iterasi
    learning_rate=0.05,      # lambat tapi stabil
    max_depth=6,             # ga terlalu dalam
    min_child_weight=5,      # minimal data di leaf
    subsample=0.8,           # pakai 80% data tiap tree
    colsample_bytree=0.8,    # pakai 80% fitur tiap tree
    gamma=1,                 # regularisasi
    tree_method='hist',      # lebih cepat di dataset besar
    n_jobs=-1,
    random_state=42
    )  # Fixed XGBRegressor
lgbm= LGBMRegressor(
   n_estimators=1000,
    learning_rate=0.05,
    num_leaves=50,            # jumlah leaf per tree
    min_child_samples=30,     # minimal data tiap leaf
    subsample=0.8,            # pakai sebagian data
    colsample_bytree=0.8,     # pakai sebagian fitur
    reg_alpha=0.5,            # regularisasi L1
    reg_lambda=0.5,           # regularisasi L2
    n_jobs=-1,
    random_state=42
    )  # Fixed n_estimators

# stacking regresor 
stack_model = StackingRegressor(
    estimators=[
        ('rf', rf), 
        ('xgb', xgb),
        ('lgbm', lgbm)
        ],
    final_estimator=Ridge(alpha=1.0),
    n_jobs=-1
)

stack_model.fit(x_train,y_train)

# voting regretor
vote_model = VotingRegressor([
    ('rf', rf),
    ('xgb', xgb),
    ('lgbm', lgbm)
], n_jobs=-1)  # opsional tapi disarankan

# print mean_absolute_error

val_pred = stack_model.predict(x_valid)
mae = mean_absolute_error(y_valid, val_pred)

# Calculate percentage metrics
mape = np.mean(np.abs((y_valid - val_pred) / y_valid)) * 100
accuracy = 100 - mape

print("\nModel Performance Metrics:")
print(f"MAE: {mae:.2f}")
print(f"MAPE: {mape:.2f}%")
print(f"Accuracy: {accuracy:.2f}%")

# Show first 5 predictions vs actual values
print("\nSample Predictions vs Actual Values:")
comparison_df = pd.DataFrame({
    'Actual': y_valid[:5],
    'Predicted': val_pred[:5],
    'Difference %': np.abs((y_valid[:5] - val_pred[:5]) / y_valid[:5] * 100)
})
print(comparison_df)

# Prediksi test.csv
test_pred = stack_model.predict(x_test_kaggle)

# Format sesuai sample_submission
submission = pd.DataFrame({
    "id": test_ids,
    "BeatsPerMinute": test_pred
})

submission.to_csv("submission.csv", index=False)
print("✅ File submission.csv siap upload ke Kaggle")

