import sys
import cv2
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout, QFrame, QSpacerItem, QSizePolicy, QDialog
)
from PyQt5.QtGui import QPixmap, QImage, QFont, QPalette, QColor, QPainter, QPen
from PyQt5.QtCore import Qt, QTimer, QSize
import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import torchvision.transforms as transforms
from PIL import Image
import requests
import json

# Load the pretrained model
weights = ResNet50_Weights.IMAGENET1K_V1
model = models.resnet50(weights=weights)
model.eval()

# Define the preprocessing steps
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=weights.transforms().mean, std=weights.transforms().std),
])

# Load ImageNet class labels
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
try:
    response = requests.get(LABELS_URL)
    response.raise_for_status()
    class_idx = json.loads(response.content)
except requests.RequestException as e:
    print(f"Error fetching ImageNet labels: {e}")
    class_idx = None

# Function to classify the image
def classify_image(image_path):
    if class_idx is None:
        return "Class labels could not be loaded.", None, None
    
    try:
        image = Image.open(image_path).convert('RGB')
    except IOError as e:
        return f"Error opening image: {e}", None, None
    
    image = transform(image).unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        outputs = model(image)
        probabilities = F.softmax(outputs, dim=1)
        
    top_prob, top_class_idx = probabilities.topk(1, dim=1)
    class_name = class_idx[top_class_idx.item()]
    
    return class_name, top_class_idx.item(), top_prob.item()

class CaptureWindow(QDialog):
    def __init__(self, image_path):
        super().__init__()
        self.setWindowTitle('Captured Image')
        
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        
        # Classify and draw result on image
        self.classify_and_display(image_path)
        
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        
        self.setLayout(layout)
    
    def classify_and_display(self, image_path):
        result = classify_image(image_path)
        class_name, class_idx, prob = result
        
        # Open image using OpenCV to draw text on it
        image = cv2.imread(image_path)
        
        # Resize the image for display
        image = cv2.resize(image, (700, 700))
        
        # Convert image to QPixmap
        qImg = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qImg)
        
        # Draw classification result on the image
        painter = QPainter(pixmap)
        painter.setPen(QPen(Qt.red))
        painter.setFont(QFont('Arial', 20))
        painter.drawText(10, 30, f'Class: {class_name} (#{class_idx})')
        painter.drawText(10, 70, f'Probability: {prob:.4f}')
        painter.end()
        
        # Display the updated pixmap
        self.image_label.setPixmap(pixmap)
        
        # Resize the window based on the image size
        self.resize(pixmap.size())

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'Image Classification'
        self.width = 900
        self.height = 700
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setFixedSize(self.width, self.height)
        
        # Set main layout
        main_layout = QVBoxLayout()
        
        # Title label
        self.title_label = QLabel('Image Classification with ResNet-50', self)
        self.title_label.setFont(QFont('Arial', 20))
        self.title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.title_label)
        
        # Horizontal layout for buttons
        button_layout = QHBoxLayout()
        
        self.btnCapture = QPushButton('Capture Image from Webcam', self)
        self.btnCapture.setStyleSheet("background-color: #4CAF50; color: white; font-size: 16px; padding: 10px;")
        self.btnCapture.clicked.connect(self.capture_image)
        button_layout.addWidget(self.btnCapture)
        
        self.btnLoad = QPushButton('Load Image from Computer', self)
        self.btnLoad.setStyleSheet("background-color: #2196F3; color: white; font-size: 16px; padding: 10px;")
        self.btnLoad.clicked.connect(self.load_image)
        button_layout.addWidget(self.btnLoad)
        
        main_layout.addLayout(button_layout)
        
        # Spacer
        main_layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))
        
        # Image display label
        self.imgLabel = QLabel(self)
        self.imgLabel.setFrameShape(QFrame.Box)
        self.imgLabel.setLineWidth(2)
        self.imgLabel.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.imgLabel)
        
        # Set main layout
        self.setLayout(main_layout)
        
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(20)
    
    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.display_image(frame)
    
    def capture_image(self):
        ret, frame = self.cap.read()
        if ret:
            image_path = 'captured_image.jpg'
            cv2.imwrite(image_path, frame)
            self.show_capture_window(image_path)
        else:
            self.show_capture_window("Error: Could not capture image")
    
    def load_image(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Choose an Image", "", "Images (*.png *.jpg *.jpeg *.bmp)", options=options)
        if fileName:
            self.show_capture_window(fileName)
    
    def show_capture_window(self, image_path):
        capture_window = CaptureWindow(image_path)
        capture_window.exec_()
    
    def display_image(self, frame):
        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)  # 1 denotes flipping around y-axis
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytesPerLine = ch * w
        qImg = QImage(frame.data, w, h, bytesPerLine, QImage.Format_RGB888)
        self.imgLabel.setPixmap(QPixmap.fromImage(qImg).scaled(self.imgLabel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
    
    def resizeEvent(self, event):
        # Adjust the webcam feed size when the main window is resized
        if self.cap.isOpened():
            self.width = event.size().width()
            self.height = event.size().height()
            self.setFixedSize(self.width, self.height)
    
    def closeEvent(self, event):
        self.cap.release()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # Setting a darker palette
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Highlight, QColor(142, 45, 197).lighter())
    palette.setColor(QPalette.HighlightedText, Qt.black)
    
    app.setPalette(palette)
    
    ex = App()
    ex.show()
    sys.exit(app.exec_())
