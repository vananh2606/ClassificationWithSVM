import os
import cv2
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from modelSVC import normalize_data, transform_image


def load_and_preprocess_image(image_path):
    """Load và tiền xử lý ảnh đầu vào"""
    # Đọc ảnh grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Không thể đọc ảnh: {image_path}")

    return image


def display_image_and_prediction(x, prediction, probability=None):
    """Hiển thị ảnh gốc và kết quả dự đoán"""
    # Đọc ảnh gốc
    plt.figure(figsize=(6, 3))

    # Hiển thị ảnh gốc
    plt.imshow(x, cmap="gray")

    if probability is not None:
        plt.title(f"Dự đoán: {prediction}\nĐộ tin cậy: {probability:.2f}%")
    else:
        plt.title(f"Dự đoán: {prediction}")

    plt.axis("off")
    plt.show()


def test_single_image(model_path, image_path):
    """Test một ảnh đơn lẻ với model đã train"""
    try:
        # Load model
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        # Load scaler đã được fit trong quá trình training
        try:
            with open("TrainModel/SVC/ModelSVC/Model/scaler_no_pca.pkl", "rb") as f:
                scaler = pickle.load(f)
        except:
            print(
                "Không tìm thấy file scaler.pkl. Vui lòng đảm bảo đã lưu scaler trong quá trình training."
            )

        # Load và xử lý ảnh
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        x = transform_image(image)
        x = normalize_data(np.array([x]), scaler)

        # Dự đoán
        class_index = model.predict(x)[0]

        # Lấy xác suất dự đoán nếu model hỗ trợ
        try:
            probability = model.predict_proba(image)[0][class_index] * 100
        except:
            probability = None

        # Hiển thị kết quả
        display_image_and_prediction(
            scaler.inverse_transform(x).reshape(28, 28), class_index, probability
        )

        return class_index, probability

    except Exception as e:
        print(f"Lỗi khi test ảnh: {str(e)}")
        return None, None


def test_all_images_in_folder(model_path, test_folder):
    """Test tất cả ảnh trong folder test"""
    # Load model
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    results = []

    # Duyệt qua từng thư mục con (0-9)
    for label in sorted(os.listdir(test_folder)):
        label_path = os.path.join(test_folder, label)
        if os.path.isdir(label_path):
            print(f"\nTesting images for digit {label}:")

            # Duyệt qua từng ảnh trong thư mục
            for image_name in sorted(os.listdir(label_path)):
                image_path = os.path.join(label_path, image_name)
                prediction, probability = test_single_image(model_path, image_path)

                if prediction is not None:
                    correct = int(label) == prediction
                    results.append(
                        {
                            "image": image_name,
                            "true_label": label,
                            "predicted": prediction,
                            "probability": probability,
                            "correct": correct,
                        }
                    )

                    # In kết quả
                    status = "✓" if correct else "✗"
                    prob_str = (
                        f" ({probability:.2f}%)" if probability is not None else ""
                    )
                    print(
                        f"{status} {image_name}: Thực tế={label}, Dự đoán={prediction}{prob_str}"
                    )

                    # Dừng một chút để người dùng có thể xem kết quả
                    plt.pause(1)

    # Tính độ chính xác tổng thể
    accuracy = sum(r["correct"] for r in results) / len(results)
    print(f"\nĐộ chính xác tổng thể: {accuracy:.2%}")

    return results


if __name__ == "__main__":
    # Đường dẫn đến model và folder test
    MODEL_PATH = "TrainModel/SVC/ModelSVC/Model/svm_model_mnist_no_pca.pkl"
    TEST_FOLDER = "TrainModel/Dataset/testing"

    # Test tất cả ảnh trong folder
    results = test_all_images_in_folder(MODEL_PATH, TEST_FOLDER)
