# Somali Celebrity Face Identification System

A Django-based web application that identifies Somali celebrities and public figures from an uploaded image or a live camera capture. The system integrates advanced computer vision pipelines with machine learning classifiers to detect faces, extract deep embeddings, verify identity, and present a full biographical summary to the user.

---

## 🚀 Key Features & Completed Steps
* **Dual Input Modes:** Supports standard file uploads (`Choose File`) and dynamic live webcam streaming via the browser's Camera API.
* **Seamless Form Integration:** Live camera captures are converted into JPEG Blobs/Files on the fly, feeding directly into the existing backend upload pipeline without breaking compatibility.
* **Rigorous Verification:** Combines an SVM classifier with explicit cosine-distance metric checks against known database embeddings to prevent false positives.
* **Rich Data Presentation:** Displays the processed image, predicted identity, model confidence score, and a comprehensive biographical record summary.
* **Modern UI/UX:** Styled with a clean blue-and-white theme, completely mobile-responsive, featuring real-time local captured-image previews.
* **Robust Error Handling:** Features empty-upload prevention, 0-byte capture filtering, and optimized in-memory image decoding using OpenCV.

---

## 🛠️ Technologies Used

### Backend & Core Logic
* **Python**
* **Django** (Web Framework & FileSystemStorage)
* **SQLite** (Database)
* **OpenCV (cv2)** & **NumPy** (Image decoding & processing)
* **Pickle** (Model serialization & rapid loading)

### Machine Learning & Computer Vision
* **MTCNN:** For highly accurate facial detection and localization.
* **DeepFace (FaceNet512):** For extracting robust 512-dimensional facial embeddings.
* **Scikit-Learn (SVM):** Main classification model mapping embeddings to celebrity labels.
* **Cosine-Distance Metrics:** Secondary validation layer checking proximity to known face representations.

### Frontend
* **HTML5, CSS3, & Vanilla JavaScript**
* **Browser Camera API** (`navigator.mediaDevices.getUserMedia`)
* **Canvas API** (For extracting frames from video streams)

---

## 📊 Model Data & Coverage
The underlying biometric database is fully mapped and optimized for the following metrics:
* **Face Embeddings stored:** 413
* **Recognized Classes (People):** 21 distinct individuals
* **Biography Coverage:** 100% complete (All 21 labels mapped with zero missing entries).
* **Notable Classes Include:** Hassan Sheikh Mohamud, Axmed Macalin Fiqi, Cabdirahmaan Odowaa, Sheikh Mustafa Haji Isma'il, Sheikh Sharif Sheikh Ahmed, Sharma Boy, Mo Farah, Ramla Ali, among others.

---

## ⚙️ How The System Works

```mermaid
graph TD
    A[User Opens UI] --> B{Choose Input Method}
    B -->|Upload File| C[Select Image File]
    B -->|Live Camera| D[Start Webcam Stream]
    D --> E[Capture Photo via Canvas API]
    E --> F[Convert JPEG Blob to File Object]
    F --> G[Inject into celebrity_image Input]
    C --> H[Click Analyze Profile Box]
    G --> H
    H --> I[Django Saves Image & Reads Bytes via OpenCV]
    I --> J[Convert BGR to RGB]
    J --> K[MTCNN Detects Faces & Selects Highest Confidence]
    K --> L[DeepFace FaceNet512 Generates 512-D Embedding]
    L --> M[SVM Classifies Identity & Cosine Distance Evaluated]
    M --> N{Confidence & Distance Checks Pass?}
    N -->|Yes| O[Display Image, Name, Confidence, and Biography Summary]
    N -->|No| P[Return Unrecognized / Error Handling]
