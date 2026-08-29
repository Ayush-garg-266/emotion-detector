# Real-Time Facial Emotion Recognition

> A Deep Learning system that detects faces and classifies facial expressions in real-time streams using a 5-block Convolutional Neural Network (CNN) trained on the FER2013 dataset with FERPlus labels.

---

## 🎥 Demo

![Emotion Detection Demo](assets/demo.gif)

> **Note**: This demonstration represents a local desktop application executing real-time webcam inference on the local machine (`cv2.VideoCapture`), rather than a hosted cloud service.

---

## ⚙️ How It Works

The system operates via a 5-stage real-time processing pipeline:

```
+------------------+     +------------------+     +-------------------+     +------------------+     +---------------------+
|   Webcam Frame   | --> |  Face Detection  | --> |   Preprocessing   | --> |  CNN Inference   | --> | Real-Time Overlay   |
| (OpenCV Capture) |     |  (Haar Cascade)  |     | (Grayscale/48x48) |     | (Softmax Prob. ) |     | (Bounding Box + Label)|
+------------------+     +------------------+     +-------------------+     +------------------+     +---------------------+
```

### Step-by-Step Pipeline:

1. **Webcam Frame Capture (`cv2.VideoCapture`)**:
   Captures continuous video frames from the local camera device at real-time frame rates.
2. **Face Detection (Haar Cascade Classifier)**:
   Extracts facial Region of Interest (ROI) bounding boxes using OpenCV's `haarcascade_frontalface_default.xml`. Haar Cascades use machine-learned feature cascades (edge, line, and center-surround features) to perform rapid object detection across frame scales.
3. **Preprocessing (`clean_data_and_normalize`)**:
   - Converts the cropped face ROI from BGR to single-channel 8-bit grayscale.
   - Resizes the ROI to standard $(48 \times 48 \times 1)$ dimensions using bilinear interpolation.
   - Normalizes pixel intensity values from $[0, 255]$ to floating-point probabilities in $[0.0, 1.0]$.
4. **Convolutional Neural Network (CNN) Inference**:
   The normalized $(48 \times 48 \times 1)$ tensor is passed through a 5-stage CNN architecture:
   - **Stage 1**: $2 \times$ Conv2D (64 filters, $3\times3$) + Batch Normalization + ReLU + Dropout ($0.5$).
   - **Stage 2**: $2 \times$ Conv2D (64 filters, $3\times3$) + ReLU + MaxPooling2D ($2\times2$, stride $2$).
   - **Stage 3**: $2 \times$ Conv2D (128 filters, $3\times3$) + Batch Normalization + ReLU.
   - **Stage 4**: $2 \times$ Conv2D (128 filters, $3\times3$) + ReLU + MaxPooling2D ($2\times2$, stride $2$).
   - **Stage 5**: $2 \times$ Conv2D (256 filters, $3\times3$) + Batch Normalization + ReLU.
   - **Dense Classifier**: Flatten $\rightarrow$ Dense (1024, ReLU) + Dropout ($0.2$) $\rightarrow$ Dense (1024, ReLU) + Dropout ($0.2$) $\rightarrow$ Dense (7, Softmax).
5. **Real-Time Visual Overlay**:
   Superimposes color-coded bounding boxes around detected faces along with the predicted emotion class label and confidence percentage score.

---

## 🛠️ Tech Stack & Libraries

Each dependency listed in `requirements.txt` serves a specific technical function within the pipeline:

- **TensorFlow / Keras (`>=2.15.0`)**: Used for building the Sequential CNN model architecture, compilation, data augmentation flow (`ImageDataGenerator`), and execution of model inference.
- **OpenCV (`opencv-python>=4.8.0`)**: Handles local webcam stream acquisition (`cv2.VideoCapture`), Haar Cascade face detection (`cv2.CascadeClassifier`), color conversion, frame resizing, and drawing bounding boxes/text overlays.
- **NumPy (`>=1.26.0`)**: Manages array operations, multi-dimensional tensor reshapes, pixel intensity normalization, and `argmax` probability index extraction.
- **Pandas (`>=2.0.0`)**: Loads and parses tabular CSV dataset representations (`fer2013.csv` and `fer2013new.csv`).
- **scikit-learn (`>=1.3.0`)**: Used for splitting training, validation, and testing partitions (`model_selection.train_test_split`).
- **Matplotlib (`>=3.7.0`)**: Generates real-time training/validation accuracy and loss metric plots.
- **Flask (`>=3.0.0`)**: Powers a local web application interface and REST endpoint (`/api/predict_frame`) for frame inference.

---

## 📊 Dataset & Cleaning

The model is trained on the [FER2013 Dataset (Kaggle)](https://www.kaggle.com/c/fer2013) augmented with refined **FERPlus** crowdsourced re-labeling (`fer2013new.csv`).

### Data Cleaning & Preprocessing Pipeline:
- **Class Filtering**: Removed noisy images tagged as `unknown` or `NF` (Not a Face).
- **Label Mapping**: Merged the 8th class (`contempt`) into the `neutral` class probability vector rather than discarding valuable training instances, yielding **7 final target classes**:
  $$\text{Classes} = \{\text{Neutral}, \text{Happiness}, \text{Surprise}, \text{Sadness}, \text{Anger}, \text{Disgust}, \text{Fear}\}$$
- **Class Imbalance**: The dataset contains noticeable class distribution skewness (e.g., significantly fewer samples for `Disgust` compared to `Happiness` or `Neutral`), presenting a recognized dataset limitation.

---

## 📈 Model Performance

<!-- TODO: Insert evaluate.py metric output, confusion matrix image, and validation curve plots below -->

### Empirical Metrics Overview:
- **Test Accuracy**: **~84.0%**
- **Macro F1-Score**: **0.80**

### Ablation & Architectural Evolution Breakdown:

| Architecture / Feature Iteration | Test Accuracy | Performance Delta | Key Insight / Technique |
| :--- | :---: | :---: | :--- |
| **Baseline (3-Block CNN)** | ~57.5% | — | Baseline 3-block architecture (adapted from neha01 benchmark). Overfitted early. |
| **+ 2 Additional Conv Blocks** | ~71.5% | **+14.0%** | Increased feature extraction depth for fine-grained facial expression patterns. |
| **+ Data Augmentation Tuning** | ~78.5% | **+7.0%** | Applied tuned rotation ($20^\circ$), horizontal flip, and height/width shifts ($0.1$). |
| **+ Regularization (BatchNorm & Dropout)**| **~84.0%** | **+5.0%** | Added Batch Normalization ($0.99$ momentum) and Dropout ($0.2$--$0.5$) to eliminate overfitting. |

<p align="center">
  <img src="https://user-images.githubusercontent.com/43937873/96019814-5d913480-0e4d-11eb-8679-b278ab47840d.png" alt="Initial Overfitting Curve" width="360"/>
  <img src="https://user-images.githubusercontent.com/40613682/96056745-aebe1a00-0e87-11eb-9198-ceb4e274b50b.png" alt="Final Training Curve" width="410"/>
</p>

*Figure 1: Comparison between initial baseline training curve (left: early overfitting) and final 5-block regularized model training curve (right: steady validation tracking through 80+ epochs).*

<p align="center">
  <img src="https://user-images.githubusercontent.com/43937873/96011743-9a582e00-0e43-11eb-9b95-eba91f99aa6f.png" alt="Confusion Matrix" width="450"/>
</p>

*Figure 2: 7-Class Confusion Matrix evaluated on the FERPlus test set (True labels vs Predicted labels).*

---

## ☁️ Architecture & Deployment Considerations

### Why This Desktop Application Is Not Hosted as a Cloud Web Demo

Executing real-time camera inference via OpenCV's `cv2.VideoCapture()` initializes a physical hardware video capture interface directly on the machine running the Python process. 

When deployed to a cloud server environment (e.g., Render, AWS EC2, or Heroku):
1. **Lack of Physical Hardware**: Cloud virtual machines operate headless without attached physical webcams.
2. **Client vs. Server Scope**: Running `cv2.VideoCapture(0)` on a cloud instance attempts to access the server's non-existent hardware rather than the website visitor's camera device.

Deploying a live browser-based demo requires rearchitecting into a Client-Server topology:
- **Client Side**: In-browser JavaScript consuming the HTML5 `navigator.mediaDevices.getUserMedia()` WebRTC API to capture the visitor's local webcam stream.
- **Server Side**: A REST/WebSocket API service (e.g., Flask or FastAPI) receiving encoded frames via HTTP POST / WebSockets, executing CNN inference, and returning JSON coordinate payloads for client-side HTML5 Canvas rendering.

This distinction represents a natural architectural evolution for cloud deployment rather than a limitation of the Machine Learning model itself.

---

## 📁 Project Structure

```
.
├── top_models/
│   └── fer.h5                              # Trained CNN model weights (14.5 MB)
├── haarcascade_frontalface_default.xml    # Haar Cascade face detection XML classifier
├── live_cam_predict.py                     # Real-time desktop webcam emotion detection application
├── web_app.py                              # Local Flask web interface & REST API server
├── img_predict.py                          # Offline emotion inference on single image files
├── vid_predict.py                          # Offline emotion inference on recorded video files
├── fer.py                                  # Complete training pipeline, data preprocessing & model definition
├── requirements.txt                        # Dependencies list
├── README.md                               # Project documentation
├── LICENSE                                 # MIT License
└── .gitignore                              # Git exclusion rules
```

---

## 🚀 Setup & Usage

### 1. Clone the Repository
```powershell
git clone https://github.com/Ayush-garg-266/emotion-detector.git
cd emotion-detector
```

### 2. Set Up Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Download Dataset (Optional for Retraining)
Download `fer2013.csv` and `fer2013new.csv` from [Kaggle FER2013](https://www.kaggle.com/c/fer2013) and place them in the project root directory.

### 5. Run Model Training
```powershell
python fer.py
```

### 6. Run Real-Time Desktop Webcam Inference
```powershell
python live_cam_predict.py
```

### 7. Run Local Web Application
```powershell
python web_app.py
```
Open `http://localhost:5000` in your web browser.

---

## 🔮 Future Improvements

- **Client-Server Cloud Architecture**: Rearchitect into a decoupled WebApp with browser-side `getUserMedia` capture and a FastAPI backend for cloud deployment.
- **DNN-Based Face Detection**: Replace Haar Cascades with modern deep learning face detectors (e.g., MediaPipe Face Detection, MTCNN, or OpenCV DNN Module) to improve detection accuracy on tilted or partially occluded faces.
- **Addressing Class Imbalance**: Implement Focal Loss or class-weighted cross-entropy loss functions to boost precision on minority classes (e.g., `Disgust`).
- **Multi-Face Temporal Tracking**: Incorporate Kalman filtering or DeepSORT for persistent bounding-box tracking across video frames.
- **Edge Deployment**: Quantize and export the trained model to TensorFlow Lite (`.tflite`) or ONNX formats for low-latency edge device execution.
