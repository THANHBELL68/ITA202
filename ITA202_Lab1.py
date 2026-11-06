# ITA202 - Lab 1: Tổng quan Học máy và cài đặt môi trường
# Bài 1 -> Bài 4

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # bỏ dòng này nếu chạy trên máy cá nhân/Colab và muốn hiện cửa sổ ảnh
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_digits
from sklearn.linear_model import LinearRegression

# =====================================================================
# BÀI 1: Làm quen với Numpy và Pandas
# =====================================================================
print("=" * 60)
print("BÀI 1: NUMPY VÀ PANDAS")
print("=" * 60)

# 1.1 Mảng Numpy 1 chiều chứa các số từ 1 đến 10
arr = np.arange(1, 11)
print("Mảng Numpy:", arr)
print("Shape:", arr.shape, "| Kiểu dữ liệu:", arr.dtype)

# 1.2 DataFrame chứa thông tin 5 sinh viên
data = {
    "Họ tên": ["Nguyễn Văn An", "Trần Thị Bình", "Lê Hoàng Cường",
               "Phạm Minh Đức", "Hoàng Thu Hà"],
    "Tuổi": [19, 20, 19, 21, 20],
    "Điểm toán": [8.5, 7.0, 9.0, 6.5, 8.0],
}
df = pd.DataFrame(data)
print("\nDataFrame sinh viên:")
print(df)

# 1.3 Thông tin tổng quan và điểm trung bình môn toán
print("\nThông tin tổng quan (df.info()):")
df.info()
print("\nThống kê mô tả (df.describe()):")
print(df.describe())
print(f"\nĐiểm trung bình môn Toán: {df['Điểm toán'].mean():.2f}")

# =====================================================================
# BÀI 2: Phân loại bài toán Machine Learning
# =====================================================================
print("\n" + "=" * 60)
print("BÀI 2: PHÂN LOẠI BÀI TOÁN ML")
print("=" * 60)

# 1. Dự đoán giá nhà dựa trên diện tích
#    -> SUPERVISED (Có giám sát - Hồi quy/Regression):
#       dữ liệu có nhãn là giá nhà (giá trị liên tục), học quan hệ diện tích -> giá.
#
# 2. Gom nhóm khách hàng dựa trên hành vi mua sắm
#    -> UNSUPERVISED (Không giám sát - Phân cụm/Clustering):
#       không có nhãn sẵn, mô hình tự tìm các nhóm khách hàng tương tự nhau.
#
# 3. Phân loại email là Spam hay không
#    -> SUPERVISED (Có giám sát - Phân loại/Classification):
#       dữ liệu có nhãn Spam / Không Spam để huấn luyện.
#
# 4. Nhận diện chữ viết tay từ hình ảnh
#    -> SUPERVISED (Có giám sát - Phân loại/Classification):
#       mỗi ảnh có nhãn là chữ số/chữ cái tương ứng (ví dụ MNIST, digits).

bai2 = pd.DataFrame({
    "Ví dụ": ["Dự đoán giá nhà theo diện tích",
              "Gom nhóm khách hàng theo hành vi mua sắm",
              "Phân loại email Spam / không Spam",
              "Nhận diện chữ viết tay từ hình ảnh"],
    "Nhóm": ["Supervised", "Unsupervised", "Supervised", "Supervised"],
    "Dạng bài toán": ["Regression", "Clustering", "Classification", "Classification"],
})
print(bai2.to_string(index=False))

# =====================================================================
# BÀI 3: Chạy thử mô hình đơn giản với Scikit-Learn
# =====================================================================
print("\n" + "=" * 60)
print("BÀI 3: LINEAR REGRESSION TRÊN TẬP IRIS")
print("=" * 60)

# Bước 1: Load dữ liệu
iris = load_iris()
X_iris, y_iris = iris.data, iris.target
print("Shape X:", X_iris.shape, "| Shape y:", y_iris.shape)
print("Tên lớp:", list(iris.target_names))

# Bước 2: Khởi tạo model
model = LinearRegression()

# Bước 3: Fit (huấn luyện)
model.fit(X_iris, y_iris)
print("Hệ số (coef_):", np.round(model.coef_, 4))
print("Hệ số chặn (intercept_):", round(model.intercept_, 4))

# Bước 4: Predict (dự đoán)
y_pred = model.predict(X_iris)
print("\n10 mẫu đầu - Nhãn thật   :", y_iris[:10])
print("10 mẫu đầu - Dự đoán thô :", np.round(y_pred[:10], 3))
print("Dự đoán làm tròn          :", np.round(y_pred[:10]).astype(int))
print(f"R^2 score: {model.score(X_iris, y_iris):.4f}")

# Nhận xét: Linear Regression trả về số thực liên tục, không phải nhãn lớp rời rạc
# (0, 1, 2), nên chỉ dùng để xem cách model hoạt động. Với Iris nên dùng mô hình
# phân loại như Logistic Regression, KNN, Decision Tree...

# =====================================================================
# BÀI 4: Quy trình ML hoàn chỉnh - Phần 1: Thu thập và kiểm tra dữ liệu
# =====================================================================
print("\n" + "=" * 60)
print("BÀI 4: THU THẬP VÀ KIỂM TRA DỮ LIỆU (DIGITS)")
print("=" * 60)

digits = load_digits()
X, y = digits.data, digits.target

print("Shape dữ liệu X:", X.shape, "-> 1797 mẫu, mỗi mẫu 64 đặc trưng (ảnh 8x8)")
print("Shape nhãn y   :", y.shape)
print("Các nhãn:", np.unique(y))
print("Số mẫu mỗi lớp:", np.bincount(y))
print("Giá trị pixel nhỏ nhất/lớn nhất:", X.min(), "/", X.max())
print("\nGiá trị mẫu đầu tiên (64 pixel):")
print(X[0])
print("Nhãn của mẫu đầu tiên:", y[0])

# Hiển thị mẫu đầu tiên dưới dạng ảnh
plt.figure(figsize=(3, 3))
plt.imshow(digits.images[0], cmap="gray_r")
plt.title(f"Mẫu đầu tiên - Nhãn: {y[0]}")
plt.axis("off")
plt.savefig("digits_sample.png", dpi=120, bbox_inches="tight")

# Hiển thị thêm 10 mẫu đầu
fig, axes = plt.subplots(2, 5, figsize=(8, 4))
for ax, img, label in zip(axes.ravel(), digits.images[:10], y[:10]):
    ax.imshow(img, cmap="gray_r")
    ax.set_title(f"Nhãn: {label}")
    ax.axis("off")
plt.tight_layout()
plt.savefig("digits_10_samples.png", dpi=120)
# plt.show()  # bỏ comment khi chạy trên máy cá nhân/Colab
print("\nĐã lưu ảnh: digits_sample.png, digits_10_samples.png")
