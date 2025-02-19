import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from model import MNIST_CNN
import torchvision.transforms as transforms


class MNISTPredictor:
    def __init__(self, model_path, device=None):
        # Xác định device (GPU/CPU)
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Load model
        self.model = MNIST_CNN().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # Định nghĩa transform
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )

    def preprocess_image(self, image):
        # Nếu ảnh là BGR, chuyển sang grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Đảm bảo kích thước ảnh là 28x28
        resized = cv2.resize(gray, (28, 28))

        # Kiểm tra xem ảnh có phải nền trắng chữ đen không
        white_pixels = np.sum(resized == 255)
        black_pixels = np.sum(resized == 0)

        # Nếu số pixel trắng nhiều hơn pixel đen (nền trắng chữ đen)
        if white_pixels > black_pixels:
            # Đảo ngược ảnh để chuyển thành nền đen chữ trắng
            resized = cv2.bitwise_not(resized)

        # Chuyển đổi sang tensor và normalize
        tensor = self.transform(resized)
        tensor = tensor.unsqueeze(0)  # Thêm batch dimension

        return tensor

    def predict(self, image):
        # Đảm bảo model ở chế độ evaluation
        self.model.eval()

        with torch.no_grad():
            # Tiền xử lý ảnh
            tensor = self.preprocess_image(image)
            tensor = tensor.to(self.device)

            # Dự đoán
            outputs = self.model(tensor)
            probabilities = torch.exp(outputs)

            # Lấy kết quả dự đoán và độ tin cậy
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item() * 100

            # Lấy xác suất cho tất cả các classes
            all_probs = probabilities[0].cpu().numpy() * 100

            return {
                "prediction": predicted_class,
                "confidence": confidence,
                "all_probabilities": all_probs,
            }

    def predict_multiple(self, images):
        """Dự đoán nhiều ảnh cùng lúc"""
        results = []
        for image in images:
            result = self.predict(image)
            results.append(result)
        return results


def load_and_preprocess_image(image_path):
    """Load và tiền xử lý ảnh đầu vào"""
    # Đọc ảnh grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Không thể đọc ảnh: {image_path}")

    return image


def display_image_and_prediction(image, prediction, probability=None):
    """Hiển thị ảnh gốc và kết quả dự đoán"""
    # Đọc ảnh gốc
    plt.figure(figsize=(10, 5))

    # Hiển thị ảnh gốc
    plt.imshow(image, cmap="gray")

    if probability is not None:
        plt.title(f"Dự đoán: {prediction}\nĐộ tin cậy: {probability:.2f}%")
    else:
        plt.title(f"Dự đoán: {prediction}")

    plt.axis("off")
    plt.show()


def test_single_image(image_path):
    """Test dự đoán trên một ảnh đơn"""
    # Khởi tạo predictor
    predictor = MNISTPredictor("TrainModel/CNN/ModelCNN/modelDL.pth")

    # Đọc ảnh
    image = load_and_preprocess_image(image_path)

    # Dự đoán
    result = predictor.predict(image)

    # In kết quả
    print(f"Dự đoán: {result['prediction']}")
    print(f"Độ tin cậy: {result['confidence']:.2f}%")

    # In xác suất cho tất cả các classes
    print("\nXác suất cho từng chữ số:")
    for digit, prob in enumerate(result["all_probabilities"]):
        print(f"Chữ số {digit}: {prob:.2f}%")

    display_image_and_prediction(image, result["prediction"], result["confidence"])


def test_folder(folder_path):
    """Test dự đoán trên tất cả ảnh trong một folder"""
    import os

    # Khởi tạo predictor
    predictor = MNISTPredictor("TrainModel/CNN/ModelCNN/modelDL.pth")

    # Đọc tất cả ảnh trong folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(folder_path, filename)
            image = load_and_preprocess_image(image_path)

            if image is not None:
                # Dự đoán
                result = predictor.predict(image)

                # In kết quả
                print(f"\nẢnh: {filename}")
                print(f"Dự đoán: {result['prediction']}")
                print(f"Độ tin cậy: {result['confidence']:.2f}%")

                display_image_and_prediction(
                    image, result["prediction"], result["confidence"]
                )
                cv2.waitKey(1000)  # Đợi 1 giây trước khi chuyển ảnh tiếp theo

    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Test một ảnh đơn
    # test_single_image("TrainModel/Dataset/testing/0/10.jpg")
    # test_single_image("src/resources/1.PNG")

    # Hoặc test cả folder
    test_folder("src/resources/")

    # Ví dụ sử dụng:
    predictor = MNISTPredictor("TrainModel/CNN/ModelCNN/modelDL.pth")

    # Đọc ảnh test
    image = cv2.imread("TrainModel/Dataset/testing/0/10.jpg")

    # Dự đoán
    result = predictor.predict(image)

    # In kết quả
    print(f"Dự đoán: {result['prediction']}")
    print(f"Độ tin cậy: {result['confidence']:.2f}%")
