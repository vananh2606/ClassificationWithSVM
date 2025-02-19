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
    QComboBox,
)
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor
from PyQt6.QtCore import Qt, QPoint

from model import MNIST_CNN, MNISTPredictor


class DrawingCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(280, 280)  # Kích thước 280x280 để dễ resize về 28x28
        self.setStyleSheet("background-color: black; border: 2px solid white;")

        # Khởi tạo biến vẽ
        self.drawing = False
        self.last_point = QPoint()
        self.image = QImage(self.size(), QImage.Format.Format_Grayscale8)
        self.image.fill(0)  # Đặt màu nền đen

    def clear_canvas(self):
        self.image.fill(0)
        self.update()

    def get_image(self):
        return self.image

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = True
            self.last_point = event.position()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = False

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.MouseButton.LeftButton) and self.drawing:
            painter = QPainter(self.image)
            painter.setPen(
                QPen(
                    QColor(255, 255, 255),
                    20,
                    Qt.PenStyle.SolidLine,
                    Qt.PenCapStyle.RoundCap,
                    Qt.PenJoinStyle.RoundJoin,
                )
            )
            painter.drawLine(self.last_point, event.position())
            self.last_point = event.position()
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(event.rect(), self.image, self.rect())


class DigitRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MNIST Digit Recognition")
        self.setGeometry(100, 100, 800, 500)

        # Initialize models as None
        self.svc_model = None
        self.svc_scaler = None
        self.cnn_predictor = None
        self.current_model = None

        # Load models
        self.load_models()

        self.setup_ui()

    def load_models(self):
        try:
            # Load SVC model
            with open(
                "TrainModel/SVC/ModelSVC/Model/svm_model_mnist_no_pca.pkl", "rb"
            ) as f:
                self.svc_model = pickle.load(f)
            with open("TrainModel/SVC/ModelSVC/Model/scaler_no_pca.pkl", "rb") as f:
                self.svc_scaler = pickle.load(f)

            # Load CNN model
            self.cnn_predictor = MNISTPredictor("TrainModel/CNN/ModelCNN/modelDL.pth")

            # Set default model
            self.current_model = "SVC"
            print("Models loaded successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not load models: {str(e)}")
            sys.exit(1)

    def setup_ui(self):
        # Widget trung tâm
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Layout chính
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(30)  # Khoảng cách giữa các layout
        main_layout.setContentsMargins(20, 20, 20, 20)  # Margin cho layout chính

        # Model selection combo box
        model_layout = QHBoxLayout()
        model_label = QLabel("Select Model:")
        model_label.setStyleSheet(
            """
            QLabel {
                font-size: 14px;
                font-weight: bold;
                color: #2c3e50;
            }
        """
        )

        self.model_combo = QComboBox()
        self.model_combo.addItems(["SVC Model", "CNN Model"])
        self.model_combo.setStyleSheet(
            """
            QComboBox {
                background-color: white;
                border: 2px solid #3498db;
                border-radius: 5px;
                padding: 5px;
                min-width: 150px;
            }
            QComboBox:hover {
                border-color: #2980b9;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: url(down_arrow.png);
                width: 12px;
                height: 12px;
            }
        """
        )
        self.model_combo.currentTextChanged.connect(self.change_model)

        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        model_layout.addStretch()

        # Layout bên trái cho vẽ
        left_layout = QVBoxLayout()
        left_layout.setSpacing(15)  # Khoảng cách giữa các widget
        left_layout.addLayout(model_layout)

        # Title cho phần vẽ
        draw_title = QLabel("Vẽ số cần nhận dạng")
        draw_title.setStyleSheet(
            """
            QLabel {
                font-size: 18px;
                font-weight: bold;
                color: #2c3e50;
                padding: 5px;
            }
        """
        )
        draw_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(draw_title)

        # Canvas vẽ
        self.canvas = DrawingCanvas()
        left_layout.addWidget(self.canvas, alignment=Qt.AlignmentFlag.AlignCenter)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)  # Khoảng cách giữa các nút

        # Style cho buttons
        button_style = """
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #2473a6;
            }
        """

        self.clear_button = QPushButton("Xóa")
        self.clear_button.setStyleSheet(button_style)
        self.clear_button.clicked.connect(self.clear_drawing)
        button_layout.addWidget(self.clear_button)

        self.predict_button = QPushButton("Nhận dạng")
        self.predict_button.setStyleSheet(
            button_style.replace("#3498db", "#27ae60")
            .replace("#2980b9", "#219a52")
            .replace("#2473a6", "#1e8449")
        )
        self.predict_button.clicked.connect(self.predict_drawing)
        button_layout.addWidget(self.predict_button)

        self.load_button = QPushButton("Chọn ảnh")
        self.load_button.setStyleSheet(
            button_style.replace("#3498db", "#e67e22")
            .replace("#2980b9", "#d35400")
            .replace("#2473a6", "#ba4a00")
        )
        self.load_button.clicked.connect(self.load_image)
        button_layout.addWidget(self.load_button)

        left_layout.addLayout(button_layout)

        # Layout bên phải cho kết quả
        right_layout = QVBoxLayout()
        right_layout.setSpacing(15)

        # Title cho phần kết quả
        result_title = QLabel("Kết quả nhận dạng")
        result_title.setStyleSheet(
            """
            QLabel {
                font-size: 18px;
                font-weight: bold;
                color: #2c3e50;
                padding: 5px;
            }
        """
        )
        result_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(result_title)

        # Label hiển thị ảnh đã xử lý
        self.processed_label = QLabel()
        self.processed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.processed_label.setMinimumSize(300, 300)
        self.processed_label.setStyleSheet(
            """
            QLabel {
                background-color: black;
                border: 2px solid #3498db;
                border-radius: 5px;
            }
        """
        )
        right_layout.addWidget(self.processed_label)

        # Label hiển thị kết quả
        self.result_label = QLabel("Vui lòng vẽ số hoặc chọn ảnh để nhận dạng")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setStyleSheet(
            """
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #2c3e50;
                padding: 10px;
                background-color: #ecf0f1;
                border-radius: 5px;
            }
        """
        )
        right_layout.addWidget(self.result_label)

        # Thêm layouts vào layout chính
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

    def change_model(self, model_name):
        self.current_model = "SVC" if "SVC" in model_name else "CNN"
        # Clear current results
        self.clear_drawing()

    def clear_drawing(self):
        self.canvas.clear_canvas()
        self.processed_label.clear()
        self.processed_label.setText("Ảnh đã xử lý")
        self.result_label.setText("Kết quả dự đoán: ")

    def predict_drawing(self):
        # Get image from canvas
        canvas_image = self.canvas.get_image()
        buffer = canvas_image.bits().asarray(canvas_image.sizeInBytes())
        img_array = np.frombuffer(buffer, dtype=np.uint8).reshape(
            (canvas_image.height(), canvas_image.width())
        )

        if self.current_model == "SVC":
            prediction, processed_image = self.predict_svc(img_array)
        else:
            prediction, processed_image = self.predict_cnn(img_array)

        # Display results
        self.display_processed_image(processed_image)
        self.display_prediction(prediction)

    def transform_image(self, image):
        X = cv2.resize(image, (28, 28)).flatten() / 255.0
        return X

    def normalize_data(self, images, scaler=None):
        X_scaled = scaler.transform(images)
        return X_scaled

    # def process_and_predict(self, image):
    #     # Nếu ảnh là RGB, chuyển sang grayscale
    #     if len(image.shape) == 3:
    #         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #     else:
    #         gray = image

    #     # Kiểm tra xem ảnh có phải nền trắng chữ đen không
    #     white_pixels = np.sum(gray == 255)
    #     black_pixels = np.sum(gray == 0)

    #     # Nếu số pixel trắng nhiều hơn pixel đen (nền trắng chữ đen)
    #     if white_pixels > black_pixels:
    #         # Đảo ngược ảnh để chuyển thành nền đen chữ trắng
    #         processed = cv2.bitwise_not(gray)
    #     else:
    #         processed = gray

    #     x = self.transform_image(processed)
    #     x = self.normalize_data(np.array([x]), self.scaler)

    #     # Dự đoán
    #     prediction = self.model.predict(x)[0]

    #     return prediction, processed

    def predict_svc(self, image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Check if image has white background
        white_pixels = np.sum(gray == 255)
        black_pixels = np.sum(gray == 0)

        if white_pixels > black_pixels:
            processed = cv2.bitwise_not(gray)
        else:
            processed = gray

        # Transform and normalize
        x = self.transform_image(processed)
        x = self.normalize_data(np.array([x]), self.svc_scaler)
        self.current_features = x

        # Predict
        prediction = self.svc_model.predict(x)[0]
        return prediction, processed

    def predict_cnn(self, image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Get prediction using CNN predictor
        result = self.cnn_predictor.predict(gray)
        self.current_prediction = result

        return result["prediction"], gray

    def display_processed_image(self, processed_image):
        h, w = processed_image.shape
        scale = min(self.processed_label.width() / w, self.processed_label.height() / h)
        new_w, new_h = int(w * scale), int(h * scale)

        processed_image = cv2.resize(processed_image, (new_w, new_h))
        q_image = QImage(
            processed_image.data, new_w, new_h, new_w, QImage.Format.Format_Grayscale8
        )
        self.processed_label.setPixmap(QPixmap.fromImage(q_image))

    def display_prediction(self, prediction):
        if self.current_model == "SVC":
            try:
                proba = self.svc_model.predict_proba(self.current_features)
                confidence = proba[0][prediction] * 100
                self.result_label.setText(
                    f"Prediction (SVC): {prediction}\nConfidence: {confidence:.2f}%"
                )
            except Exception as e:
                self.result_label.setText(f"Prediction (SVC): {prediction}")
        else:
            confidence = self.current_prediction["confidence"]
            self.result_label.setText(
                f"Prediction (CNN): {prediction}\nConfidence: {confidence:.2f}%"
            )

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Chọn ảnh", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )

        if file_name:
            try:
                image = cv2.imread(file_name)
                if image is None:
                    raise Exception("Không thể đọc ảnh")

                # Xử lý ảnh và dự đoán
                if self.current_model == "SVC":
                    prediction, processed_image = self.predict_svc(image)
                else:
                    prediction, processed_image = self.predict_cnn(image)

                # Vẽ ảnh lên canvas
                self.draw_image_on_canvas(processed_image)

                # Hiển thị ảnh đã xử lý
                self.display_processed_image(processed_image)

                # Hiển thị kết quả
                self.display_prediction(prediction)

            except Exception as e:
                QMessageBox.critical(self, "Lỗi", f"Lỗi khi xử lý ảnh: {str(e)}")

    def draw_image_on_canvas(self, image):
        # Resize về kích thước canvas
        resized = cv2.resize(image, (self.canvas.width(), self.canvas.height()))

        # Chuyển thành QImage
        h, w = resized.shape
        q_image = QImage(resized.data, w, h, w, QImage.Format.Format_Grayscale8)

        # Vẽ lên canvas
        self.canvas.image = q_image
        self.canvas.update()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DigitRecognitionApp()
    window.show()
    sys.exit(app.exec())
