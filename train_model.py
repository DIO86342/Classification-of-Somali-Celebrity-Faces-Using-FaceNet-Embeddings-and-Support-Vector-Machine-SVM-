import os
import cv2
import pickle
import numpy as np

from mtcnn import MTCNN
from deepface import DeepFace

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

DATASET_PATH = r"C:\Users\hp\Downloads\last-new-celebrities"

detector = MTCNN()

X = []
y = []

print("[INFO] Processing dataset...")

for person_name in os.listdir(DATASET_PATH):

    person_dir = os.path.join(DATASET_PATH, person_name)

    if not os.path.isdir(person_dir):
        continue

    print(f"Processing: {person_name}")

    for image_name in os.listdir(person_dir):

        image_path = os.path.join(person_dir, image_name)

        img = cv2.imread(image_path)

        if img is None:
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        detections = detector.detect_faces(rgb)

        if len(detections) == 0:
            continue

        best_face = max(detections, key=lambda x: x["confidence"])

        x1, y1, w, h = best_face["box"]

        x1 = max(0, x1)
        y1 = max(0, y1)

        face = rgb[y1:y1+h, x1:x1+w]

        if face.size == 0:
            continue

        try:
            # Convert RGB back to BGR for DeepFace (it expects BGR)
            face_bgr = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
            
            embedding = DeepFace.represent(
                img_path=face_bgr,
                model_name="Facenet512",
                detector_backend="skip",
                enforce_detection=False
            )[0]["embedding"]

            X.append(embedding)
            y.append(person_name)

        except Exception as e:
            print("Skipped:", e)

X = np.array(X)
y = np.array(y)

print("Total faces:", len(X))

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

clf = SVC(
    kernel="rbf",  # Changed from linear to RBF for better multiclass separation
    probability=True,
    C=1.0,  # Regularization parameter
    gamma='scale'  # Gamma parameter for RBF kernel
)

clf.fit(X_train, y_train)
pred = clf.predict(X_test)

acc = accuracy_score(y_test, pred)

print("Accuracy:", acc * 100, "%")

# Store both embeddings and labels for distance-based verification
known_faces = {
    "embeddings": X,
    "labels": y  # Keep track of labels for each embedding
}

with open("known_faces.pkl", "wb") as f:
    pickle.dump(known_faces, f)

with open("somali_celebrity_svm4.pkl", "wb") as f:
    pickle.dump(clf, f)
print("DONE")