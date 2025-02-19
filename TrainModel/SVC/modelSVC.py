import os
import cv2
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt


def load_datasets_from_folder(folder_path):
    X = []
    y = []

    # Duyệt qua các thư mục con (0-9)
    for label in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label)
        if os.path.isdir(label_path):
            # Duyệt qua từng ảnh trong thư mục
            for image_name in os.listdir(label_path):
                image_path = os.path.join(label_path, image_name)
                # Đọc ảnh bằng cv2
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    X.append(transform_image(image))
                    y.append(int(label))
    return np.array(X), y


def transform_image(image):
    X = cv2.resize(image, (28, 28)).flatten() / 255.0
    return X


def fit_scaler(X_train):
    scaler = StandardScaler()
    scaler.fit(X_train)
    with open("TrainModel/SVC/ModelSVC/Model/scaler_no_pca.pkl", "wb") as f:
        pickle.dump(scaler, f)
    return scaler


def normalize_data(images, scaler: StandardScaler = None):
    X_scaled = scaler.transform(images)
    return X_scaled


def shuffle_data(X, y):
    X, y = shuffle(X, y, random_state=42)
    return X, y


def visualize_samples(X, y, num_samples=10, title="Sample Images"):
    """
    Hiển thị ảnh mẫu theo từng nhãn (0-9).
    """
    unique_labels = np.unique(y)  # Các nhãn có trong dữ liệu
    plt.figure(figsize=(30, 6))
    plt.suptitle("Sample Images by Label", fontsize=16)

    for i, label in enumerate(unique_labels):
        # Chọn num_samples ảnh có nhãn tương ứng
        indices = np.where(y == label)[0][:num_samples]
        for j, idx in enumerate(indices):
            plt.subplot(len(unique_labels), num_samples, i * num_samples + j + 1)
            img = X[idx].reshape(28, 28)  # Reshape về ảnh 28x28
            plt.imshow(img, cmap="gray")
            plt.axis("off")

    plt.tight_layout()
    plt.show()


def pca_components(X_train_scaled, X_test_scaled, n_components=100):
    """Giảm chiều dữ liệu với PCA"""
    pca = PCA(n_components=n_components)  # Giữ lại 100 components
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    return X_train_pca, X_test_pca


def visualize_samples_pca(X, y, num_samples=10, title="Sample Images PCA"):
    """
    Hiển thị ảnh mẫu theo từng nhãn (0-9).
    """
    n_components = 10 * 10
    unique_labels = np.unique(y)  # Các nhãn có trong dữ liệu
    plt.figure(figsize=(30, 6))
    plt.suptitle("Sample Images by Label", fontsize=16)

    for i, label in enumerate(unique_labels):
        # Chọn num_samples ảnh có nhãn tương ứng
        indices = np.where(y == label)[0][:num_samples]
        for j, idx in enumerate(indices):
            plt.subplot(len(unique_labels), num_samples, i * num_samples + j + 1)
            img = X[idx].reshape(10, 10)  # Reshape về ảnh 10x10
            plt.imshow(img, cmap="gray")
            plt.axis("off")

    plt.tight_layout()
    plt.show()


def visualize_pca_2_components(X_train_pca, X_test_pca, y_train, y_test, n_sample=100):
    plt.subplot(1, 2, 1)
    plt.scatter(X_train_pca[:, 0], X_train_pca[:n_sample, 1], c=y_train, cmap="viridis")
    plt.title("Normalized Training Data")
    plt.subplot(1, 2, 2)
    plt.scatter(X_test_pca[:, 0], X_test_pca[:n_sample, 1], c=y_test, cmap="viridis")
    plt.title("Normalized Testing Data")
    plt.show()


def visualize_pca_3_components(X_train_pca, X_test_pca, y_train, y_test, n_sample=100):
    fig = plt.figure(figsize=(12, 5))

    # Biểu đồ 3D cho tập train
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.scatter(
        X_train_pca[:n_sample, 0],
        X_train_pca[:n_sample, 1],
        X_train_pca[:n_sample, 2],
        c=y_train[:n_sample],
        cmap="viridis",
        alpha=0.8,
    )
    ax1.set_title("3D PCA - Training Data")
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.set_zlabel("PC3")

    # Biểu đồ 3D cho tập test
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.scatter(
        X_test_pca[:n_sample, 0],
        X_test_pca[:n_sample, 1],
        X_test_pca[:n_sample, 2],
        c=y_test[:n_sample],
        cmap="viridis",
        alpha=0.8,
    )
    ax2.set_title("3D PCA - Testing Data")
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.set_zlabel("PC3")

    plt.show()


def train_svm(X_train, y_train, kernel="linear", C=1.0):
    """Huấn luyện mô hình SVM với kernel tùy chọn"""
    model = SVC(kernel=kernel, C=C)
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model,
    X_test,
    y_test,
    filename="TrainModel/SVC/ModelSVC/Evaluation/model_evaluation_no_pca.txt",
):
    """Đánh giá mô hình SVM"""
    y_pred = model.predict(X_test)

    # In độ chính xác
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy Score: {acc:.2f}")

    # In báo cáo phân loại
    class_report = classification_report(y_test, y_pred)
    print("\nClassification Report:")
    print(class_report)

    # In ma trận nhầm lẫn
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(conf_matrix)

    # Ghi kết quả vào file
    with open(filename, "w") as f:
        f.write(f"Accuracy Score: {acc:.2f}\n\n")
        f.write("Classification Report:\n")
        f.write(class_report + "\n\n")
        f.write("Confusion Matrix:\n")
        np.savetxt(f, conf_matrix, fmt="%d")

    print(f"\nKết quả đã được lưu vào {filename}")


def visualize_confusion_matrix(y_test, y_pred):
    # Vẽ confusion matrix
    plt.figure(figsize=(10, 10))
    cm = confusion_matrix(y_test, y_pred)
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xticks(np.arange(10))
    plt.yticks(np.arange(10))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    # Thêm giá trị số vào các ô
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                horizontalalignment="center",
                verticalalignment="center",
            )

    plt.savefig("TrainModel/SVC/ModelSVC/Evaluation/confusion_matrix_no_pca.png")
    plt.show()


if __name__ == "__main__":
    # Load dữ liệu training và testing
    print("Loading training data...")
    X_train, y_train = load_datasets_from_folder("TrainModel/Dataset/training")
    print("Loading testing data...")
    X_test, y_test = load_datasets_from_folder("TrainModel/Dataset/testing")

    scaler: StandardScaler = fit_scaler(X_train)

    # Chuẩn hóa dữ liệu
    X_train_scaled = normalize_data(X_train, scaler)
    X_test_scaled = normalize_data(X_test, scaler)

    # Xáo trộn dữ liệu
    X_train_scaled, y_train = shuffle_data(X_train_scaled, y_train)
    X_test_scaled, y_test = shuffle_data(X_test_scaled, y_test)

    # Trực quan hóa dữ liệu
    visualize_samples(X_train_scaled, y_train)

    # Giảm chiều dữ liệu
    # X_train_pca, X_test_pca = pca_components(X_train_scaled, X_test_scaled, 100)

    # X_train_pca, X_test_pca = pca_components(X_train_scaled, X_test_scaled, 2)
    # visualize_pca_2_components(X_train_pca, X_test_pca, y_train, y_test)

    # X_train_pca, X_test_pca = pca_components(X_train_scaled, X_test_scaled, 3)
    # visualize_pca_3_components(X_train_pca, X_test_pca, y_train, y_test)

    # Trực quan hóa các thành phần PCA
    # visualize_samples_pca(X_train_pca, y_train)

    # Train mô hình SVM với kernel tuyến tính
    svm_model = train_svm(X_train_scaled, y_train, kernel="linear", C=1.0)

    # Save model with pickle
    with open("TrainModel/SVC/ModelSVC/Model/svm_model_mnist_no_pca.pkl", "wb") as f:
        pickle.dump(svm_model, f)

    # Load model with pickle
    with open("TrainModel/SVC/ModelSVC/Model/svm_model_mnist_no_pca.pkl", "rb") as f:
        loaded_svm_model = pickle.load(f)

    # Đánh giá mô hình
    evaluate_model(loaded_svm_model, X_test_scaled, y_test)

    # Vẽ confusion matrix
    y_pred = loaded_svm_model.predict(X_test_scaled)
    visualize_confusion_matrix(y_test, y_pred)
