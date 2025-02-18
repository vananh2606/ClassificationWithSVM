import sys
import cv2
import numpy as np
import pickle
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QMessageBox,
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt


class DigitRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MNIST Digit Recognition")
        self.setGeometry(100, 100, 800, 500)

        # Load model và scaler
        try:
            with open(
                "TrainModel/ModelSVC/Model/svm_model_mnist_no_pca.pkl", "rb"
            ) as f:
                self.model = pickle.load(f)
            with open("TrainModel/ModelSVC/Model/scaler_no_pca.pkl", "rb") as f:
                self.scaler = pickle.load(f)
            print("Model và Scaler đã được load thành công!")
        except Exception as e:
            QMessageBox.critical(
                self, "Lỗi", f"Không thể load model hoặc scaler: {str(e)}"
            )
            sys.exit(1)

        self.setup_ui()

    def setup_ui(self):
        # Widget trung tâm
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Layout chính
        main_layout = QHBoxLayout(central_widget)

        # Layout bên trái cho ảnh
        left_layout = QVBoxLayout()

        # Label hiển thị ảnh gốc
        self.original_label = QLabel("Ảnh gốc")
        self.original_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.original_label.setMinimumSize(300, 300)
        self.original_label.setStyleSheet("border: 2px solid black")
        left_layout.addWidget(self.original_label)

        # Nút chọn file
        self.load_button = QPushButton("Chọn ảnh")
        self.load_button.clicked.connect(self.load_image)
        left_layout.addWidget(self.load_button)

        # Layout bên phải cho kết quả
        right_layout = QVBoxLayout()

        # Label hiển thị ảnh đã xử lý
        self.processed_label = QLabel("Ảnh đã xử lý")
        self.processed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.processed_label.setMinimumSize(300, 300)
        self.processed_label.setStyleSheet("border: 2px solid black")
        right_layout.addWidget(self.processed_label)

        # Label hiển thị kết quả
        self.result_label = QLabel("Kết quả dự đoán: ")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        right_layout.addWidget(self.result_label)

        # Thêm layouts vào layout chính
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

    def load_image(self):
        # Mở dialog chọn file
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Chọn ảnh", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )

        if file_name:
            try:
                # Đọc và xử lý ảnh
                image = cv2.imread(file_name)
                if image is None:
                    raise Exception("Không thể đọc ảnh")

                # Hiển thị ảnh gốc
                self.display_original_image(image)

                # Xử lý ảnh và dự đoán
                prediction, processed_image = self.process_and_predict(image)

                # Hiển thị ảnh đã xử lý
                self.display_processed_image(processed_image)

                # Hiển thị kết quả
                self.display_prediction(prediction)

            except Exception as e:
                QMessageBox.critical(self, "Lỗi", f"Lỗi khi xử lý ảnh: {str(e)}")

    def display_original_image(self, image):
        # Chuyển từ BGR sang RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape

        # Scale ảnh để vừa với label
        scale = min(self.original_label.width() / w, self.original_label.height() / h)
        new_w, new_h = int(w * scale), int(h * scale)

        # Resize ảnh
        rgb_image = cv2.resize(rgb_image, (new_w, new_h))

        # Chuyển sang QImage và hiển thị
        q_image = QImage(
            rgb_image.data, new_w, new_h, new_w * ch, QImage.Format.Format_RGB888
        )
        self.original_label.setPixmap(QPixmap.fromImage(q_image))

    def transform_image(self, image):
        x = cv2.resize(image, (28, 28)).flatten() / 255.0
        return x

    def normalize_data(self, images, s=None):
        x_scaled = s.transform(images)
        return x_scaled

    def process_and_predict(self, image):
        # Chuyển sang grayscale
        processed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Kiểm tra xem ảnh có phải nền trắng chữ đen không
        white_pixels = np.sum(processed == 255)
        black_pixels = np.sum(processed == 0)

        # Nếu số pixel trắng nhiều hơn pixel đen (nền trắng chữ đen)
        if white_pixels > black_pixels:
            # Đảo ngược ảnh để chuyển thành nền đen chữ trắng
            processed = cv2.bitwise_not(processed)

        display_processed = processed.copy()

        x = self.transform_image(processed)
        x = self.normalize_data(np.array([x]), self.scaler)

        # Dự đoán
        prediction = self.model.predict(x)[0]

        return prediction, display_processed

    def display_processed_image(self, processed_image):
        # Scale ảnh lên để dễ nhìn
        h, w = processed_image.shape
        scale = min(self.processed_label.width() / w, self.processed_label.height() / h)
        new_w, new_h = int(w * scale), int(h * scale)

        # Resize ảnh
        processed_image = cv2.resize(processed_image, (new_w, new_h))

        # Chuyển sang QImage và hiển thị
        q_image = QImage(
            processed_image.data, new_w, new_h, new_w, QImage.Format.Format_Grayscale8
        )
        self.processed_label.setPixmap(QPixmap.fromImage(q_image))

    def display_prediction(self, prediction):
        try:
            confidence = (
                self.model.predict_proba(self.scaler.transform(self.current_features))[
                    0
                ][prediction]
                * 100
            )
            self.result_label.setText(
                f"Kết quả dự đoán: {prediction}\nĐộ tin cậy: {confidence:.2f}%"
            )
        except:
            self.result_label.setText(f"Kết quả dự đoán: {prediction}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DigitRecognitionApp()
    window.show()
    sys.exit(app.exec())
