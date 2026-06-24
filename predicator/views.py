import os
import cv2
import pickle
import numpy as np
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage

# Global variables for models (lazy loading)
detector = None
embedder = None
svm_classifier = None
known_embeddings = None
known_norm_embeddings = None
known_labels = None
class_distance_thresholds = None

MIN_CLASS_DISTANCE_THRESHOLD = 0.22
MAX_CLASS_DISTANCE_THRESHOLD = 0.45
CLASS_DISTANCE_MARGIN = 0.05
SVM_CONFIDENCE_THRESHOLD = 45.0
SVM_PROBABILITY_GAP_THRESHOLD = 10.0
NEAREST_CLASS_MARGIN_THRESHOLD = 0.03
STRONG_NEAREST_DISTANCE_THRESHOLD = 0.18
STRONG_NEAREST_MARGIN_THRESHOLD = 0.05


def l2_normalize(values):
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def build_class_distance_thresholds(embeddings, labels):
    """Estimate a cosine-distance acceptance threshold for every known class."""
    thresholds = {}

    for label in np.unique(labels):
        indexes = np.where(labels == label)[0]
        same_class_distances = []

        for index in indexes:
            other_indexes = indexes[indexes != index]
            if other_indexes.size == 0:
                continue

            distances = 1 - np.dot(embeddings[other_indexes], embeddings[index])
            same_class_distances.append(float(np.min(distances)))

        if same_class_distances:
            raw_threshold = np.percentile(same_class_distances, 90) + CLASS_DISTANCE_MARGIN
        else:
            raw_threshold = MIN_CLASS_DISTANCE_THRESHOLD

        thresholds[label] = float(np.clip(
            raw_threshold,
            MIN_CLASS_DISTANCE_THRESHOLD,
            MAX_CLASS_DISTANCE_THRESHOLD
        ))

    return thresholds

def load_models():
    """Load detector, SVM classifier, and embeddings on first use"""
    global detector, embedder, svm_classifier, known_embeddings, known_norm_embeddings
    global known_labels, class_distance_thresholds
    
    if detector is None:
        from mtcnn import MTCNN
        detector = MTCNN()

    if embedder is None:
        from deepface import DeepFace

        # ==========================================
        # STANDARD 512-DIMENSIONAL FACENET EMBEDDER
        # ==========================================
        class GenuineFaceNetEmbedder:
            def __init__(self):
                print("[INFO] DeepFace standard representation engine active.")

            def embeddings(self, face_pixels):
                # DeepFace expects BGR image data when feeding raw NumPy arrays.
                # The incoming cropped face is in RGB format from OpenCV conversion.
                face_bgr = cv2.cvtColor(face_pixels, cv2.COLOR_RGB2BGR)
                representations = DeepFace.represent(
                    img_path=face_bgr,
                    model_name="Facenet512",
                    enforce_detection=False,
                    detector_backend="skip"
                )
                
                embedding_list = representations[0]["embedding"]
                features = np.array(embedding_list).reshape(1, -1)
                return features

        embedder = GenuineFaceNetEmbedder()
    
    if svm_classifier is None:
        MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'somali_celebrity_svm4.pkl')
        with open(MODEL_PATH, 'rb') as f:
            svm_classifier = pickle.load(f)
    
    if known_embeddings is None:
        KNOWN_PATH = os.path.join(
            os.path.dirname(__file__),
            'models',
            'known_faces.pkl'
        )
        with open(KNOWN_PATH, 'rb') as f:
            known_data = pickle.load(f)
        known_embeddings = np.asarray(known_data['embeddings'], dtype=np.float32)
        known_norm_embeddings = l2_normalize(known_embeddings)
        known_labels = np.asarray(known_data['labels'])
        class_distance_thresholds = build_class_distance_thresholds(
            known_norm_embeddings,
            known_labels
        )
# CELEBRITY KNOWLEDGE DATABASE MAPPING
CELEBRITY_DATA = {
    "Axmed Macalin Fiqi": {
        "name": "Axmed Macalin Fiqi",
        "description": "Siyaasi iyo diblomaasi Soomaaliyeed oo xilal kala duwan ka soo qabtay dowladda federaalka. Waxaa lagu yaqaan ka qeybqaadashada arrimaha amniga, diblomaasiyadda, iyo maamulka guud."
    },
    "Cabdirahmaan Odowaa": {
        "name": "Cabdirahmaan Maxamed Xuseen Odowaa",
        "description": "Siyaasi Soomaaliyeed oo ka tirsanaa hoggaanka dowladda federaalka, kana soo shaqeeyay arrimaha gudaha, dib u heshiisiinta, iyo dhismaha hay'adaha dowliga ah."
    },
    "Cabdulqaadir Maxamed Abshir (nadaara)": {
        "name": "Cabdulqaadir Maxamed Abshir (Nadaara)",
        "description": "Shaqsi Soomaaliyeed oo caan ku ah bulshada dhexdeeda, gaar ahaan muuqaalada, madadaalada, iyo ka dhex muuqashada baraha bulshada."
    },
    "Cumar Kiris": {
        "name": "Cumar Kiris",
        "description": "Siyaasi Soomaaliyeed oo kasoo muuqday saaxadda siyaasadda iyo maamulka, lana xiriira hawlaha adeegga bulshada iyo arrimaha dowladnimada."
    },
    "Derisle Abdi": {
        "name": "Derisle Abdi",
        "description": "Shaqsi Soomaaliyeed oo ka dhex muuqda bulshada iyo warbaahinta casriga ah. Xogtiisa waxaa lagu kaydiyay nidaamka aqoonsiga wajiyada ee mashruucan."
    },
    "Happy Ahmed": {
        "name": "Happy Ahmed",
        "description": "Abuure muuqaal iyo shaqsi Soomaaliyeed oo caan ka ah baraha bulshada, kuna yaqaan nuxur madadaalo iyo ka qeybgalka dhaqanka dhalinyarada."
    },
    "Hassan Sheekh Mohmoud": {
        "name": "Hassan Sheikh Mohamud",
        "description": "Hoggaamiye siyaasadeed iyo aqoonyahan Soomaaliyeed oo laba jeer loo doortay madaxweynaha Jamhuuriyadda Federaalka Soomaaliya. Waxaa lagu yaqaan shaqadiisa dowlad-dhiska, waxbarashada, iyo siyaasadda qaranka."
    },
    "Ilkacase Qays": {
        "name": "Ilkacase Qays",
        "description": "Fanaan Soomaaliyeed oo caan ku ah heesaha casriga ah iyo codkiisa gaarka ah. Waxaa uu ka mid yahay fanaaniinta si weyn looga dhageysto bulshada Soomaaliyeed."
    },
    "Jamal Osman": {
        "name": "Jamal Osman",
        "description": "Suxufi iyo soo saare warbaahineed oo Soomaali ah, kana soo shaqeeyay warbixinno iyo barnaamijyo ka hadla arrimaha Soomaaliya, siyaasadda, iyo bulshada."
    },
    "Maxamed Cabdi Sharmaarke (Sharma Boy)": {
        "name": "Maxamed Cabdi Sharmaarke (Sharma Boy)",
        "description": "Fanaan iyo rapper Soomaaliyeed oo caan ku ah heesaha hip-hop-ka Soomaaliga, laxanno firfircoon, iyo fariimo la xiriira dhalinyarada iyo nolosha casriga ah."
    },
    "Moh farah": {
        "name": "Mo Farah",
        "description": "Orodyahan asal ahaan Soomaali ah oo ku guuleystay billado caalami ah, gaar ahaan tartamada orodka masaafada dheer. Waxaa lagu tiriyaa halyeeyada ciyaaraha fudud."
    },
    "Mohamed Abdulahi farmajo": {
        "name": "Mohamed Abdullahi Mohamed (Farmaajo)",
        "description": "Siyaasi Soomaaliyeed oo xilka madaxweynaha Soomaaliya hayay 2017 ilaa 2022, horeyna u soo noqday ra'iisul wasaare. Waxaa lagu yaqaan doorkiisa siyaasadda qaranka iyo dib u habeynta hay'adaha."
    },
    "Omar Artan": {
        "name": "Omar Artan",
        "description": "Shaqsi Soomaaliyeed oo caan ka ah bulshada iyo baraha bulshada. Nidaamkan wuxuu u aqoonsadaa mid ka mid ah profiles-ka lagu tababaray model-ka."
    },
    "Ramla Ali": {
        "name": "Ramla Ali",
        "description": "Feeryahanad iyo model asal ahaan Soomaali ah oo heer caalami ah ka gaartay ciyaaraha feerka. Waxay sidoo kale caan ku tahay dhiirrigelinta haweenka iyo matalaadda bulshada Soomaaliyeed."
    },
    "Salah Sanaag": {
        "name": "Salah Sanaag",
        "description": "Fanaan Soomaaliyeed oo lagu yaqaan heeso iyo bandhigyo ka dhex muuqda fanka Soomaalida. Waxaa uu ka mid yahay magacyada ku jira database-ka wajiyada ee nidaamkan."
    },
    "Sheekh Siciid Raage": {
        "name": "Sheekh Siciid Raage",
        "description": "Caalim iyo daaci Soomaaliyeed oo caan ku ah muxaadarooyin diini ah, wacyigelin, iyo ka hadalka arrimaha bulshada ee ku saleysan fahamka diinta Islaamka."
    },
    "Sheikh Cabdirashiid Sheekh Cali Sufi": {
        "name": "Sheikh Cabdirashiid Sheekh Cali Sufi",
        "description": "Qari iyo caalim Soomaaliyeed oo si weyn loogu yaqaan qur'aan akhriska iyo codkiisa macaan. Waxaa uu ka mid yahay culimada Soomaaliyeed ee caan ka ah dunida Islaamka."
    },
    "Sheikh Mustafa Haji Isma'il": {
        "name": "Sheikh Mustafa Haji Isma'il",
        "description": "Daaci iyo caalim Soomaaliyeed oo caan ku ah muxaadarooyinka, tafsiirka, iyo wacyigelinta bulshada. Waxaa si weyn looga dhageystaa Soomaalida gudaha iyo dibadda."
    },
    "Somali Gamer": {
        "name": "Somali Gamer",
        "description": "Abuure nuxur Soomaaliyeed oo ku caan baxay ciyaaraha elektaroonigga ah iyo muuqaalada internet-ka. Waxaa uu ka mid yahay dadka ka dhex muuqda dhaqanka digital-ka Soomaalida."
    },
    "cabdiraxmaan cabdishakuur": {
        "name": "Cabdiraxmaan Cabdishakuur Warsame",
        "description": "Siyaasi Soomaaliyeed, diblomaasi, iyo falanqeeye siyaasadeed oo door muuqda ku leh doodaha dowladnimada, dimuqraadiyadda, iyo arrimaha qaranka Soomaaliya."
    },
    "sheikh Sharif Sheikh Ahmed": {
        "name": "Sheikh Sharif Sheikh Ahmed",
        "description": "Siyaasi Soomaaliyeed oo hore u soo noqday madaxweynaha Soomaaliya. Waxaa lagu yaqaan kaalintiisa siyaasadda, dib u heshiisiinta, iyo hoggaaminta xilli muhiim ah oo dalka soo maray."
    },
    "Hassan_Sheikh": {
        "name": "Hassan Sheikh Mohamud",
        "description": "Hoggaamiye siyaasadeed iyo aqoonyahan Soomaaliyeed oo laba jeer loo doortay madaxweynaha Jamhuuriyadda Federaalka Soomaaliya. Waxaa lagu yaqaan shaqadiisa dowlad-dhiska, waxbarashada, iyo siyaasadda qaranka."
    },
    "Mohamed_Farmaajo": {
        "name": "Mohamed Abdullahi Mohamed (Farmaajo)",
        "description": "Siyaasi Soomaaliyeed oo xilka madaxweynaha Soomaaliya hayay 2017 ilaa 2022, horeyna u soo noqday ra'iisul wasaare. Waxaa lagu yaqaan doorkiisa siyaasadda qaranka iyo dib u habeynta hay'adaha."
    },
    "Sharma_Boy": {
        "name": "Maxamed Cabdi Sharmaarke (Sharma Boy)",
        "description": "Fanaan iyo rapper Soomaaliyeed oo caan ku ah heesaha hip-hop-ka Soomaaliga, laxanno firfircoon, iyo fariimo la xiriira dhalinyarada iyo nolosha casriga ah."
    },
    "Suldaan_Seeraar": {
        "name": "Suldaan Seeraar",
        "description": "Fanaan Soomaaliyeed oo caan ku ah heesaha casriga ah iyo cod macaan oo si weyn looga dhageysto bulshada Soomaaliyeed."
    },
    "Unknown": {
        "name": "Unknown Identity",
        "description": "Wajiga la baaray lama jaanqaadin qof ku jira database-ka maxalliga ah ee model-ka lagu tababaray."
    }
}

# The index function remains exactly the same as written before...
def index(request):
    context = {}
    if request.method == 'POST' and request.FILES.get('celebrity_image'):
        load_models()  # Load models only when the upload form is submitted
        # Save the uploaded file to the media folder
        uploaded_file = request.FILES['celebrity_image']
        if uploaded_file.size == 0:
            context['error'] = 'The uploaded image was empty. Please capture the photo again or choose a valid JPG or PNG.'
            return render(request, 'index.html', context)

        fs = FileSystemStorage()
        filename = fs.save(uploaded_file.name, uploaded_file)
        
        # Paths for template viewing and system calculation
        uploaded_file_url = fs.url(filename)
        absolute_file_path = fs.path(filename)

        try:
            # Load image using OpenCV and shift color profile to RGB
            image_bytes = np.fromfile(absolute_file_path, dtype=np.uint8)
            img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
            if img is None:
                context['error'] = 'Could not read the uploaded image file. Ensure it is a valid JPG or PNG.'
                return render(request, 'index.html', context)
                
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Detect face
            detections = detector.detect_faces(rgb_img)

            if len(detections) == 0:
                context['error'] = 'Face detection layer failed. The image might be too dark, blurry, or misaligned.'
                return render(request, 'index.html', context)

            # Isolate primary target face box
            best_face = max(detections, key=lambda i: i['confidence'])
            x, y, w, h = best_face['box']
            
            # CRITICAL SAFETY FIX: Prevent negative or out-of-bounds array coordinates
            x, y = max(0, x), max(0, y)
            
            # Crop the face array matrix safely
            cropped_face = rgb_img[y:y+h, x:x+w]

            # CRITICAL SAFETY FIX: Verify the cropped section actually contains image data
            if cropped_face.size == 0 or cropped_face.shape[0] == 0 or cropped_face.shape[1] == 0:
                context['error'] = 'Face detection isolated an invalid image bounding region. Please try another photo.'
                return render(request, 'index.html', context)

            # 1. Extract proper features
            embeddings = embedder.embeddings(cropped_face)

            embedding = embeddings.flatten().astype(np.float32)
            norm_embedding = l2_normalize(embedding)

            # Compare FaceNet512 embeddings with cosine distance after L2 normalization.
            distances = 1 - np.dot(known_norm_embeddings, norm_embedding)
            nearest_index = int(np.argmin(distances))
            min_distance = float(distances[nearest_index])
            closest_label = known_labels[nearest_index]
            class_threshold = class_distance_thresholds.get(
                closest_label,
                MAX_CLASS_DISTANCE_THRESHOLD
            )

            other_class_distances = distances[known_labels != closest_label]
            nearest_other_distance = (
                float(np.min(other_class_distances))
                if other_class_distances.size
                else float('inf')
            )
            nearest_class_margin = nearest_other_distance - min_distance

            probabilities = svm_classifier.predict_proba(embeddings)[0]
            ranked_indexes = np.argsort(probabilities)[::-1]
            top_index = int(ranked_indexes[0])
            second_index = int(ranked_indexes[1]) if ranked_indexes.size > 1 else top_index
            predicted_class = svm_classifier.classes_[top_index]
            svm_confidence = float(probabilities[top_index] * 100)
            probability_gap = float(
                (probabilities[top_index] - probabilities[second_index]) * 100
            )

            distance_ok = min_distance <= class_threshold
            classifiers_agree = predicted_class == closest_label
            confidence_ok = svm_confidence >= SVM_CONFIDENCE_THRESHOLD
            gap_ok = probability_gap >= SVM_PROBABILITY_GAP_THRESHOLD
            nearest_margin_ok = nearest_class_margin >= NEAREST_CLASS_MARGIN_THRESHOLD
            strong_nearest_ok = (
                min_distance <= min(class_threshold, STRONG_NEAREST_DISTANCE_THRESHOLD)
                and nearest_class_margin >= STRONG_NEAREST_MARGIN_THRESHOLD
            )

            print(
                "FaceNet nearest:",
                closest_label,
                f"distance={min_distance:.3f}",
                f"threshold={class_threshold:.3f}",
                f"margin={nearest_class_margin:.3f}"
            )
            print(
                "SVM top:",
                predicted_class,
                f"confidence={svm_confidence:.1f}%",
                f"gap={probability_gap:.1f}%"
            )

            if distance_ok and (
                (
                    classifiers_agree
                    and confidence_ok
                    and (gap_ok or nearest_margin_ok)
                )
                or strong_nearest_ok
            ):
                predicted_class_folder = predicted_class if classifiers_agree else closest_label
                confidence_score = svm_confidence
                print(f"MATCH: {predicted_class_folder}")
            else:
                predicted_class_folder = "Unknown"
                confidence_score = 0
                print("UNKNOWN FACE: failed open-set recognition checks")
            # Pull mapping details matching predicted label
            celebrity_profile = CELEBRITY_DATA.get(predicted_class_folder, {
                "name": predicted_class_folder.replace("_", " "),
                "description": "No biography entry found for this classified folder target."
            })

            # Ship context attributes back to UI layout
            context['image_url'] = uploaded_file_url
            context['name'] = celebrity_profile['name']
            context['description'] = celebrity_profile['description']
            context['confidence'] = f"{confidence_score:.1f}%"
        except Exception as e:
            context['error'] = f'Pipeline execution failure: {str(e)}'

    return render(request, 'index.html', context)

