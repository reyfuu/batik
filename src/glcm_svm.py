import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from src.lbp_extractor import extract_lbp_features

try:
    from skimage.feature import graycomatrix, graycoprops
except ImportError:
    from skimage.feature import greycomatrix as graycomatrix, greycoprops as graycoprops

def extract_glcm_features(img_gray):
    """Ekstrak fitur tekstur GLCM dari gambar grayscale."""
    # Kuantisasi ke 32 level gray -> matriks GLCM lebih ringan & tidak sparse
    img_quantized = (img_gray / 8).astype(np.uint8)

    distances = [1, 2, 4]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    glcm = graycomatrix(img_quantized, distances=distances, angles=angles, levels=32, symmetric=True, normed=True)

    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    feature_vector = []
    
    for prop in properties:
        values = graycoprops(glcm, prop)
        feature_vector.extend(values.flatten())
        
    return np.array(feature_vector)

FEATURE_SIZE = 224  # semua gambar disamakan ke ukuran dataset training (DATASETv7 = 224x224)


def extract_combined_features(img_gray):
    """Gabungkan fitur GLCM + LBP jadi satu vektor.

    WAJIB resize ke FEATURE_SIZE dulu: GLCM (jarak 1-4 px) & LBP (radius 1 px) dihitung
    dalam satuan piksel, jadi tekstur foto 3000px != gambar 224px. Tanpa ini fitur beda
    skala dari training -> akurasi jatuh ke tebak acak. Titik tunggal yang dilewati semua
    caller (train & predict).
    """
    img_gray = cv2.resize(img_gray, (FEATURE_SIZE, FEATURE_SIZE), interpolation=cv2.INTER_AREA)
    glcm_feat = extract_glcm_features(img_gray)
    lbp_feat = extract_lbp_features(img_gray)
    return np.concatenate([glcm_feat, lbp_feat])

def load_dataset(base_dir, split):
    """Muat gambar, ekstrak fitur GLCM + LBP gabungan, kumpulkan label."""
    split_dir = os.path.join(base_dir, split)
    if not os.path.exists(split_dir):
        raise ValueError(f"Directory {split_dir} does not exist. Please generate dataset first.")
        
    categories = ["parang", "kawung", "megamendung"]
    features = []
    labels = []
    
    print(f"Mengekstrak fitur GLCM + LBP dari dataset '{split}'...")
    for label_id, cat in enumerate(categories):
        cat_dir = os.path.join(split_dir, cat)
        files = [f for f in os.listdir(cat_dir) if f.endswith('.png')]
        
        for file in files:
            filepath = os.path.join(cat_dir, file)
            img = cv2.imread(filepath)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            feat = extract_combined_features(gray)
            features.append(feat)
            labels.append(label_id)
            
    return np.array(features), np.array(labels)

def train_and_evaluate(base_dir="data", model_path="models"):
    """Latih & bandingkan SVM vs KNN pada fitur GLCM+LBP."""
    os.makedirs(model_path, exist_ok=True)

    try:
        X_train, y_train = load_dataset(base_dir, "train")
        X_test, y_test = load_dataset(base_dir, "test")
    except ValueError as e:
        print(f"Error: {e}")
        return None, None
        
    print(f"Jumlah data latih: {len(X_train)} sampel (Panjang fitur: {X_train.shape[1]})")
    print(f"Jumlah data uji: {len(X_test)} sampel")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Melatih model Support Vector Machine (SVM) dengan RBF Kernel...")
    svm = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42)
    svm.fit(X_train_scaled, y_train)
    y_pred_svm = svm.predict(X_test_scaled)
    acc_svm = accuracy_score(y_test, y_pred_svm)

    print("Melatih model K-Nearest Neighbors (KNN)...")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    y_pred_knn = knn.predict(X_test_scaled)
    acc_knn = accuracy_score(y_test, y_pred_knn)
    
    print("\n" + "="*50)
    print(f"HASIL EVALUASI MODEL TRADISIONAL (GLCM + LBP)")
    print("="*50)
    print(f"Akurasi SVM: {acc_svm * 100:.2f}%")
    print(f"Akurasi KNN: {acc_knn * 100:.2f}%")
    print("\nLaporan Klasifikasi SVM:")
    categories = ["Parang", "Kawung", "Megamendung"]
    print(classification_report(y_test, y_pred_svm, target_names=categories))
    print("="*50 + "\n")

    svm_file = os.path.join(model_path, "svm_model.pkl")
    knn_file = os.path.join(model_path, "knn_model.pkl")
    scaler_file = os.path.join(model_path, "scaler.pkl")
    
    with open(svm_file, "wb") as f:
        pickle.dump(svm, f)
    with open(knn_file, "wb") as f:
        pickle.dump(knn, f)
    with open(scaler_file, "wb") as f:
        pickle.dump(scaler, f)
        
    print(f"Model SVM disimpan ke {svm_file}")
    print(f"Model KNN disimpan ke {knn_file}")
    print(f"Scaler disimpan ke {scaler_file}")
    
    return svm, knn, scaler

def train_from_folders(train_dir, model_path="models", exts=(".jpg", ".jpeg", ".png")):
    """Latih SVM (GLCM+LBP) pada dataset folder (subfolder = kelas).

    Simpan svm_batik.pkl, scaler_batik.pkl, classes.pkl. Return (svm, scaler, classes).
    """
    classes = sorted(d for d in os.listdir(train_dir)
                     if os.path.isdir(os.path.join(train_dir, d)))
    X, y = [], []
    for idx, cls in enumerate(classes):
        cdir = os.path.join(train_dir, cls)
        for f in os.listdir(cdir):
            if not f.lower().endswith(exts):
                continue
            img = cv2.imread(os.path.join(cdir, f))
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            X.append(extract_combined_features(gray))
            y.append(idx)

    X, y = np.array(X), np.array(y)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    svm = SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=42)
    svm.fit(Xs, y)

    os.makedirs(model_path, exist_ok=True)
    pickle.dump(svm, open(os.path.join(model_path, "svm_batik.pkl"), "wb"))
    pickle.dump(scaler, open(os.path.join(model_path, "scaler_batik.pkl"), "wb"))
    pickle.dump(classes, open(os.path.join(model_path, "classes.pkl"), "wb"))
    print(f"[SVM] Latih {len(X)} sampel, {len(classes)} kelas -> {model_path}/svm_batik.pkl")
    return svm, scaler, classes


def predict_single(image_path, model_path="models", classifier_type="svm"):
    """Prediksi satu gambar dengan model SVM atau KNN tersimpan."""
    scaler_file = os.path.join(model_path, "scaler.pkl")
    model_file = os.path.join(model_path, f"{classifier_type}_model.pkl")
    
    if not (os.path.exists(model_file) and os.path.exists(scaler_file)):
        print(f"Model {classifier_type.upper()} belum dilatih. Jalankan pelatihan terlebih dahulu.")
        return
        
    with open(model_file, "rb") as f:
        model = pickle.load(f)
    with open(scaler_file, "rb") as f:
        scaler = pickle.load(f)
        
    img = cv2.imread(image_path)
    if img is None:
        print(f"Gambar tidak ditemukan: {image_path}")
        return
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    feat = extract_combined_features(gray).reshape(1, -1)
    feat_scaled = scaler.transform(feat)
    
    pred_idx = model.predict(feat_scaled)[0]

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(feat_scaled)[0][pred_idx]
        confidence_str = f"{prob * 100:.2f}%"
    else:
        confidence_str = "N/A"
        
    categories = ["Parang", "Kawung", "Megamendung"]
    print(f"Hasil Prediksi GLCM + LBP + {classifier_type.upper()} untuk {os.path.basename(image_path)}:")
    print(f"- Motif Terdeteksi: {categories[pred_idx]}")
    print(f"- Confidence: {confidence_str}")

if __name__ == "__main__":
    train_and_evaluate()
