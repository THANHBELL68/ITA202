# -*- coding: utf-8 -*-
"""
GIAI ĐOẠN 1: Chuẩn bị dữ liệu & xây dựng mô hình cơ sở
Bài toán : Dự đoán khả năng sống sót trên tàu Titanic (PHÂN LOẠI nhị phân)
Dữ liệu  : titanic.csv (891 hành khách) - nguồn: seaborn-data (mwaskom)
Mô hình  : Logistic Regression (baseline)
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay,
                             mean_squared_error)

RANDOM_STATE = 42
sns.set_theme(style="whitegrid")

# ---------------------------------------------------------------
# 1. CHỌN VÀ KHÁM PHÁ DỮ LIỆU
# ---------------------------------------------------------------
df = pd.read_csv("data/titanic.csv")
print("=== Shape ===");    print(df.shape)
print("\n=== Info ===");   df.info()
print("\n=== Describe ==="); print(df.describe(include="all").T)
print("\n=== Giá trị khuyết thiếu ==="); print(df.isna().sum())

# Biểu đồ khám phá
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
sns.countplot(data=df, x="survived", ax=ax[0]); ax[0].set_title("Phân bố nhãn Survived")
sns.barplot(data=df, x="sex", y="survived", ax=ax[1]); ax[1].set_title("Tỉ lệ sống sót theo giới tính")
sns.barplot(data=df, x="pclass", y="survived", ax=ax[2]); ax[2].set_title("Tỉ lệ sống sót theo hạng vé")
plt.tight_layout(); plt.savefig("figures/01_eda.png", dpi=120); plt.close()

# ---------------------------------------------------------------
# 2. LÀM SẠCH DỮ LIỆU
# ---------------------------------------------------------------
# Loại cột trùng lặp / rò rỉ nhãn (alive = survived) / quá nhiều NaN (deck)
drop_cols = ["alive", "class", "who", "adult_male", "alone",
             "embark_town", "deck"]
df = df.drop(columns=drop_cols)
# (Không drop_duplicates: bộ dữ liệu không có ID, các dòng giống nhau vẫn là hành khách khác nhau)
# Embarked chỉ thiếu 2 dòng -> điền giá trị phổ biến nhất (mode = "S")
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

# Mã hóa biến phân loại
df["sex"] = df["sex"].map({"male": 0, "female": 1})
df = pd.get_dummies(df, columns=["embarked"], drop_first=True, dtype=int)

X = df.drop(columns="survived")
y = df["survived"]

# Chia train/test 80/20 (stratify giữ nguyên tỉ lệ nhãn).
# Chia TRƯỚC khi impute/scale để tránh rò rỉ dữ liệu (data leakage).
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

# Xử lý NaN: điền bằng median của tập train
medians = X_train.median()
X_train = X_train.fillna(medians)
X_test = X_test.fillna(medians)

# Xử lý outlier (IQR): cắt (clip) giá trị ở age, fare theo ngưỡng của tập train
for col in ["age", "fare"]:  # sibsp/parch là biến đếm nhỏ, không áp IQR
    q1, q3 = X_train[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = int(((X_train[col] < low) | (X_train[col] > high)).sum())
    print(f"Outlier '{col}': {n_out} giá trị ngoài [{low:.2f}, {high:.2f}] -> clip")
    X_train[col] = X_train[col].clip(low, high)
    X_test[col] = X_test[col].clip(low, high)

# Boxplot fare trước/sau khi clip
fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
sns.boxplot(x=X["fare"], ax=ax[0]); ax[0].set_title("Fare - trước khi xử lý outlier")
sns.boxplot(x=X_train["fare"], ax=ax[1]); ax[1].set_title("Fare - sau khi clip (train)")
plt.tight_layout(); plt.savefig("figures/02_outlier_fare.png", dpi=120); plt.close()

# ---------------------------------------------------------------
# 3. TIỀN XỬ LÝ: CHUẨN HÓA (StandardScaler, fit trên train)
# ---------------------------------------------------------------
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
print(f"\nTrain: {X_train_s.shape}  Test: {X_test_s.shape}")

# ---------------------------------------------------------------
# 4. HUẤN LUYỆN MÔ HÌNH CƠ SỞ: LOGISTIC REGRESSION
# ---------------------------------------------------------------
model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
model.fit(X_train_s, y_train)

# ---------------------------------------------------------------
# 5. ĐÁNH GIÁ BAN ĐẦU
# ---------------------------------------------------------------
y_pred = model.predict(X_test_s)
res = {
    "accuracy_train": accuracy_score(y_train, model.predict(X_train_s)),
    "accuracy_test": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred),
    "mse_test": mean_squared_error(y_test, y_pred),
}
print("\n=== KẾT QUẢ MÔ HÌNH CƠ SỞ (Logistic Regression) ===")
for k, v in res.items():
    print(f"{k:15s}: {v:.4f}")
json.dump(res, open("models/baseline_results.json", "w"), indent=2)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=["Chết", "Sống"]).plot(cmap="Blues")
plt.title("Confusion Matrix - Logistic Regression (Test)")
plt.tight_layout(); plt.savefig("figures/03_cm_baseline.png", dpi=120); plt.close()

# Hệ số mô hình: biến nào ảnh hưởng nhiều nhất
coef = pd.Series(model.coef_[0], index=X.columns).sort_values()
coef.plot(kind="barh", figsize=(7, 4), color="steelblue")
plt.title("Hệ số Logistic Regression (đã chuẩn hóa)")
plt.tight_layout(); plt.savefig("figures/04_coef_baseline.png", dpi=120); plt.close()
print("\nHệ số:\n", coef)
