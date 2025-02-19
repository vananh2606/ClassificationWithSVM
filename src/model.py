import torch
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        # First Convolutional Layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)

        # Second Convolutional Layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)

        # Fully Connected Layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # First Conv Layer
        x = self.conv1(x)  # 28x28 -> 28x28
        x = F.relu(x)
        x = self.pool1(x)  # 28x28 -> 14x14

        # Second Conv Layer
        x = self.conv2(x)  # 14x14 -> 14x14
        x = F.relu(x)
        x = self.pool2(x)  # 14x14 -> 7x7

        # Flatten
        x = x.view(-1, 64 * 7 * 7)

        # Fully Connected
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)


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
