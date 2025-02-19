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
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor
from PyQt6.QtCore import Qt, QPoint


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

        # Load model và scaler
        try:
            with open(
                "TrainModel/SVC/ModelSVC/Model/svm_model_mnist_no_pca.pkl", "rb"
            ) as f:
                self.model = pickle.load(f)
            with open("TrainModel/SVC/ModelSVC/Model/scaler_no_pca.pkl", "rb") as f:
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
        main_layout.setSpacing(30)  # Khoảng cách giữa các layout
        main_layout.setContentsMargins(20, 20, 20, 20)  # Margin cho layout chính

        # Layout bên trái cho vẽ
        left_layout = QVBoxLayout()
        left_layout.setSpacing(15)  # Khoảng cách giữa các widget

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

    def clear_drawing(self):
        self.canvas.clear_canvas()
        self.processed_label.clear()
        self.processed_label.setText("Ảnh đã xử lý")
        self.result_label.setText("Kết quả dự đoán: ")

    def predict_drawing(self):
        # Lấy ảnh từ canvas
        canvas_image = self.canvas.get_image()

        # Chuyển QImage thành numpy array
        buffer = canvas_image.bits().asarray(canvas_image.sizeInBytes())
        img_array = np.frombuffer(buffer, dtype=np.uint8).reshape(
            (canvas_image.height(), canvas_image.width())
        )

        # Xử lý và dự đoán
        prediction, processed_image = self.process_and_predict(img_array)

        # Hiển thị ảnh đã xử lý
        self.display_processed_image(processed_image)

        # Hiển thị kết quả
        self.display_prediction(prediction)

    def transform_image(self, image):
        X = cv2.resize(image, (28, 28)).flatten() / 255.0
        return X

    def normalize_data(self, images, scaler=None):
        X_scaled = scaler.transform(images)
        return X_scaled

    def process_and_predict(self, image):
        # Nếu ảnh là RGB, chuyển sang grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Kiểm tra xem ảnh có phải nền trắng chữ đen không
        white_pixels = np.sum(gray == 255)
        black_pixels = np.sum(gray == 0)

        # Nếu số pixel trắng nhiều hơn pixel đen (nền trắng chữ đen)
        if white_pixels > black_pixels:
            # Đảo ngược ảnh để chuyển thành nền đen chữ trắng
            processed = cv2.bitwise_not(gray)
        else:
            processed = gray

        x = self.transform_image(processed)
        x = self.normalize_data(np.array([x]), self.scaler)

        # Dự đoán
        prediction = self.model.predict(x)[0]

        return prediction, processed

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
        try:
            proba = self.model.predict_proba(self.current_features)
            confidence = proba[0][prediction] * 100
            self.result_label.setText(
                f"Kết quả dự đoán: {prediction}\nĐộ tin cậy: {confidence:.2f}%"
            )
        except Exception as e:
            print(f"Không thể tính độ tin cậy: {e}")
            self.result_label.setText(f"Kết quả dự đoán: {prediction}")

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
                prediction, processed_image = self.process_and_predict(image)

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
