# 📸 Image Processing using OpenCV (Python)

This project demonstrates various image processing techniques using **OpenCV** and **Python**.  
It includes basic operations such as image loading, resizing, color conversions, filtering, edge detection, thresholding, morphological operations, and more.

---

## 🧰 Technologies Used
- **Python 3.x**
- **OpenCV (cv2)**
- **NumPy**
- **Matplotlib** (optional for visualization)

---

## 📂 Project Structure
├── images/ # Input images
├── outputs/ # Processed results
├── src/
│ ├── basic_operations.py
│ ├── filters.py
│ ├── edge_detection.py
│ ├── thresholding.py
│ ├── morphology.py
│ └── utils.py
└── README.md


---

## ✨ Features

### ✔️ Basic Operations
- Load, display, and save images  
- Resize, rotate, crop  
- Convert to grayscale  
- Draw shapes and add text  

### ✔️ Image Filtering
- Gaussian Blur  
- Median Filter  
- Bilateral Filter  
- Custom convolution filters  

### ✔️ Edge Detection
- Sobel operator  
- Prewitt (custom)  
- Scharr  
- Canny edge detector  

### ✔️ Thresholding
- Global thresholding  
- Adaptive thresholding  
- Otsu’s binarization  

### ✔️ Morphological Operations
- Erosion  
- Dilation  
- Opening & Closing  
- Morphological Gradient  

### ✔️ Additional Modules
- Histogram equalization  
- Contour detection  
- Image segmentation  

---

## 🚀 Getting Started

### 1️⃣ Install Dependencies

pip install opencv-python numpy matplotlib
2️⃣ Clone This Repository
bash
Copier le code
git clone https://github.com/your-username/image-processing-opencv.git
cd image-processing-opencv
3️⃣ Run a Script
bash
Copier le code
python src/basic_operations.py
🧪 Example Usage
python
Copier le code
import cv2

# Load an image
img = cv2.imread("images/lena.png")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Display the result
cv2.imshow("Blurred Image", blur)
cv2.waitKey(0)
cv2.destroyAllWindows()

📘 Useful Documentation

OpenCV Docs: https://docs.opencv.org

NumPy Docs: https://numpy.org/doc



---
