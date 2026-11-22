# -*- coding: utf-8 -*-
"""
GIAI ĐOẠN 2: Tối ưu hóa & hoàn thiện dự án (Titanic - phân loại)
 6. Thử KNN, Decision Tree, Random Forest
 7. Pipeline (ColumnTransformer + mô hình)
 8. GridSearchCV + Cross-Validation (5-fold)
 9. So sánh với mô hình cơ sở
10. Lưu mô hình bằng joblib (.pkl)
"""
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay, roc_auc_score)

RANDOM_STATE = 42
sns.set_theme(style="whitegrid")

# ---------- Đọc dữ liệu & chia train/test (cùng random_state với Giai đoạn 1) ----------
df = pd.read_csv("data/titanic.csv")
features_num = ["age", "sibsp", "parch", "fare"]
features_cat = ["pclass", "sex", "embarked"]
X = df[features_num + features_cat]
y = df["survived"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

# ---------- 7. PIPELINE: tiền xử lý + mô hình ----------
# Pipeline giúp mọi bước (impute, scale, encode) chỉ fit trên train -> không rò rỉ dữ liệu
preprocess = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scaler", StandardScaler())]), features_num),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), features_cat),
])

def make_pipe(model):
    return Pipeline([("prep", preprocess), ("model", model)])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

def evaluate(name, pipe):
    p = pipe.predict(X_test)
    return {"model": name,
            "cv_acc": None,
            "train_acc": accuracy_score(y_train, pipe.predict(X_train)),
            "test_acc": accuracy_score(y_test, p),
            "precision": precision_score(y_test, p),
            "recall": recall_score(y_test, p),
            "f1": f1_score(y_test, p),
            "roc_auc": roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])}

results, best_estimators = [], {}

# ---------- Mô hình cơ sở (Logistic Regression) trong Pipeline để so sánh công bằng ----------
base = make_pipe(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)).fit(X_train, y_train)
r = evaluate("Logistic Regression (baseline)", base)
r["cv_acc"] = cross_val_score(base, X_train, y_train, cv=cv).mean()
r["best_params"] = "-"
results.append(r)

# ---------- 6 + 8. KNN / Decision Tree / Random Forest + GridSearchCV ----------
grids = {
    "KNN": (KNeighborsClassifier(),
            {"model__n_neighbors": [3, 5, 7, 9, 11, 15, 21],
             "model__weights": ["uniform", "distance"],
             "model__p": [1, 2]}),
    "Decision Tree": (DecisionTreeClassifier(random_state=RANDOM_STATE),
            {"model__max_depth": [2, 3, 4, 5, 6, 8, None],
             "model__min_samples_split": [2, 10, 20],
             "model__min_samples_leaf": [1, 5, 10],
             "model__criterion": ["gini", "entropy"]}),
    "Random Forest": (RandomForestClassifier(random_state=RANDOM_STATE),
            {"model__n_estimators": [100, 300],
             "model__max_depth": [3, 5, 8, None],
             "model__min_samples_split": [2, 5, 10],
             "model__min_samples_leaf": [1, 2, 4]}),
}
for name, (model, grid) in grids.items():
    gs = GridSearchCV(make_pipe(model), grid, cv=cv, scoring="accuracy", n_jobs=-1)
    gs.fit(X_train, y_train)
    print(f"\n[{name}] best CV acc = {gs.best_score_:.4f} | params = {gs.best_params_}")
    r = evaluate(name + " (tuned)", gs.best_estimator_)
    r["cv_acc"] = gs.best_score_
    r["best_params"] = {k.replace("model__", ""): v for k, v in gs.best_params_.items()}
    results.append(r)
    best_estimators[name] = gs.best_estimator_

# ---------- 9. SO SÁNH ----------
res = pd.DataFrame(results)
print("\n=== BẢNG SO SÁNH ===")
print(res.drop(columns="best_params").round(4).to_string(index=False))
res.to_csv("models/model_comparison.csv", index=False)

# Chọn mô hình cuối theo CV accuracy (không chọn theo test để tránh thiên lệch)
best_row = res.iloc[1:].sort_values("cv_acc", ascending=False).iloc[0]
best_name = best_row["model"].replace(" (tuned)", "")
final_model = best_estimators[best_name]
print(f"\n>>> Mô hình cuối cùng: {best_name}")

# Biểu đồ so sánh
plot_df = res.melt(id_vars="model", value_vars=["train_acc", "cv_acc", "test_acc"],
                   var_name="Chỉ số", value_name="Accuracy")
plt.figure(figsize=(10, 4.5))
sns.barplot(data=plot_df, x="model", y="Accuracy", hue="Chỉ số")
plt.ylim(0.6, 1.0); plt.xticks(rotation=15); plt.title("So sánh các mô hình (Train / CV / Test)")
plt.tight_layout(); plt.savefig("figures/05_model_comparison.png", dpi=120); plt.close()

# Confusion matrix mô hình cuối
p = final_model.predict(X_test)
ConfusionMatrixDisplay(confusion_matrix(y_test, p), display_labels=["Chết", "Sống"]).plot(cmap="Greens")
plt.title(f"Confusion Matrix - {best_name} (Test)")
plt.tight_layout(); plt.savefig("figures/06_cm_final.png", dpi=120); plt.close()

# Độ quan trọng đặc trưng (nếu mô hình hỗ trợ)
est = final_model.named_steps["model"]
if hasattr(est, "feature_importances_"):
    names = final_model.named_steps["prep"].get_feature_names_out()
    imp = pd.Series(est.feature_importances_, index=names).sort_values()
    imp.plot(kind="barh", figsize=(7, 5), color="seagreen")
    plt.title(f"Feature importance - {best_name}")
    plt.tight_layout(); plt.savefig("figures/07_feature_importance.png", dpi=120); plt.close()

# ---------- 10. ĐÓNG GÓI ----------
joblib.dump(final_model, "models/titanic_final_model.pkl")
json.dump({"final_model": best_name, "metrics": {k: float(best_row[k]) for k in
           ["cv_acc", "train_acc", "test_acc", "precision", "recall", "f1", "roc_auc"]},
           "best_params": {k: (v if v is None or isinstance(v, (int, float, str)) else str(v))
                           for k, v in best_row["best_params"].items()}},
          open("models/final_summary.json", "w"), indent=2, ensure_ascii=False)

# Kiểm tra nạp lại và dự đoán hành khách mới
loaded = joblib.load("models/titanic_final_model.pkl")
new = pd.DataFrame([{"age": 25, "sibsp": 0, "parch": 0, "fare": 80, "pclass": 1, "sex": "female", "embarked": "C"},
                    {"age": 40, "sibsp": 0, "parch": 0, "fare": 8, "pclass": 3, "sex": "male", "embarked": "S"}])
print("\nDự đoán hành khách mới (1=sống):", loaded.predict(new), loaded.predict_proba(new)[:, 1].round(3))
