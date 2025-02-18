import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_data(file_path):
    """Load dữ liệu từ file CSV"""
    df = pd.read_csv(file_path)
    X = df[["Feature 1", "Feature 2"]].values
    y = df["Label"].values
    return X, y


def normalize_data(X_train, X_test):
    """Chuẩn hóa dữ liệu bằng StandardScaler"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def visualize_normalize_data(X_train_scaled, X_test_scaled, y_train, y_test):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap="viridis")
    plt.title("Normalized Training Data")
    plt.subplot(1, 2, 2)
    plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=y_test, cmap="viridis")
    plt.title("Normalized Testing Data")
    plt.show()


def train_svm(X_train, y_train, kernel="linear", C=1.0):
    """Huấn luyện mô hình SVM với kernel tùy chọn"""
    model = SVC(kernel=kernel, C=C)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Đánh giá mô hình SVM"""
    y_pred = model.predict(X_test)

    # In độ chính xác
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy Score: {acc:.2f}")

    # In báo cáo phân loại
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # In ma trận nhầm lẫn
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


def plot_decision_boundary(model, X, y):
    """Vẽ ranh giới quyết định của SVM"""
    h = 0.02  # Kích thước bước lưới
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Decision Boundary của SVM")
    plt.show()


# Load dữ liệu
X, y = load_data("TrainModel/TestSVC/dataset_2D.csv")

# Chia tập train và test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Chuẩn hóa dữ liệu
X_train_scaled, X_test_scaled = normalize_data(X_train, X_test)

# Trực quan hóa dữ liệu
visualize_normalize_data(X_train_scaled, X_test_scaled, y_train, y_test)

# Train mô hình SVM với kernel tuyến tính
svm_model = train_svm(X_train_scaled, y_train, kernel="linear", C=1.0)

# Save model with pickle
with open("TrainModel/TestSVC/svm_model.pkl", "wb") as f:
    pickle.dump(svm_model, f)

# Load model with pickle
with open("TrainModel/TestSVC/svm_model.pkl", "rb") as f:
    loaded_svm_model = pickle.load(f)

# Đánh giá mô hình
evaluate_model(loaded_svm_model, X_test_scaled, y_test)

# Vẽ Decision Boundary
plot_decision_boundary(svm_model, X_train_scaled, y_train)
