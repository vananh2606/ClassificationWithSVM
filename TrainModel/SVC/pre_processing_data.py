import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Số điểm dữ liệu mỗi lớp
N = 100

# Trung tâm của hai lớp
mu_0 = [2, 2]
mu_1 = [6, 6]

# Ma trận hiệp phương sai (càng lớn thì dữ liệu càng phân tán)
cov = [[1.2, 0.5], [0.5, 1.2]]

# Tạo dữ liệu từ phân phối chuẩn
X0 = np.random.multivariate_normal(mu_0, cov, N)
X1 = np.random.multivariate_normal(mu_1, cov, N)

# Gán nhãn (Class 0: label=0, Class 1: label=1)
y0 = np.zeros((N, 1))  # Label 0
y1 = np.ones((N, 1))  # Label 1

# Gộp dữ liệu và nhãn
X = np.vstack((X0, X1))
y = np.vstack((y0, y1))

# Chuyển thành DataFrame
df = pd.DataFrame(np.hstack((X, y)), columns=["Feature 1", "Feature 2", "Label"])

# Lưu thành file CSV
df.to_csv("TrainModel/SVC/TestSVC/dataset_2D.csv", index=False)
print("Dữ liệu đã được lưu vào file dataset_2D.csv")

# Trực quan hóa dữ liệu
plt.figure(figsize=(8, 6))
plt.scatter(X0[:, 0], X0[:, 1], c="b", label="Class 0")
plt.scatter(X1[:, 0], X1[:, 1], c="r", label="Class 1")
plt.legend()
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Bộ dữ liệu 2 chiều với 2 class")
plt.grid()
plt.show()
