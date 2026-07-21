import os
import sys
import random

# Add current directory to path to enable importing from src
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

def print_header():
    print("\n" + "="*65)
    print("      BATIK MOTIF ANALYSIS CLI - AIPRENUR & RAVELOUX      ")
    print("="*65)
    print("  [1] Generate Synthetic Batik Dataset (Parang, Kawung, Megamendung)")
    print("  [2] Run GLCM + LBP + SVM/KNN Demo (Traditional Texture Classification)")
    print("  [3] Run CNN MobileNetV2 Demo (Deep Learning Classification)")
    print("  [4] Run Object Detection Demo (Faster R-CNN / CV Localization)")
    print("  [5] Run Speed Benchmark of 5 Models (Save chart to public/img)")
    print("  [6] Keluar / Exit")
    print("="*65)

def check_dataset_exists():
    return os.path.exists("data/train/parang") and os.path.exists("data/test/parang")

def get_random_test_image():
    if not check_dataset_exists():
        return None
    categories = ["parang", "kawung", "megamendung"]
    cat = random.choice(categories)
    test_dir = os.path.join("data", "test", cat)
    files = [f for f in os.listdir(test_dir) if f.endswith('.png')]
    if not files:
        return None
    return os.path.join(test_dir, random.choice(files))

def main():
    while True:
        print_header()
        try:
            choice = input("Pilih menu [1-6]: ").strip()
        except KeyboardInterrupt:
            print("\nKeluar dari program...")
            break
            
        if choice == '1':
            from src.generator import create_dataset
            create_dataset(base_dir="data", num_train=40, num_test=15)
            
        elif choice == '2':
            if not check_dataset_exists():
                print("\n[WARNING] Dataset belum ada! Menjalankan generator otomatis dahulu...")
                from src.generator import create_dataset
                create_dataset()
                
            from src.glcm_svm import train_and_evaluate, predict_single
            print("\n--- Memulai Klasifikasi GLCM + LBP + SVM/KNN ---")
            train_and_evaluate()
            
            # Predict a random test image
            test_img = get_random_test_image()
            if test_img:
                print("\n--- Menjalankan Uji Coba Prediksi Gambar Acak dengan SVM ---")
                predict_single(test_img, classifier_type="svm")
                print("\n--- Menjalankan Uji Coba Prediksi Gambar Acak dengan KNN ---")
                predict_single(test_img, classifier_type="knn")
            else:
                print("Tidak ada gambar uji untuk diprediksi.")
                
        elif choice == '3':
            if not check_dataset_exists():
                print("\n[WARNING] Dataset belum ada! Menjalankan generator otomatis dahulu...")
                from src.generator import create_dataset
                create_dataset()
                
            from src.cnn_mobilenet import train_and_evaluate as train_cnn, predict_single as predict_cnn
            print("\n--- Memulai Klasifikasi Deep Learning MobileNetV2 ---")
            # We run 3 epochs for quick CLI demonstration
            train_cnn(epochs=3, batch_size=16)
            
            # Predict a random test image
            test_img = get_random_test_image()
            if test_img:
                print("\n--- Menjalankan Uji Coba Prediksi Gambar Acak ---")
                predict_cnn(test_img)
            else:
                print("Tidak ada gambar uji untuk diprediksi.")
                
        elif choice == '4':
            from src.object_detection import run_detection_demo
            print("\n--- Memulai Demo Object Detection (Faster R-CNN / CV) ---")
            run_detection_demo()
            
        elif choice == '5':
            if not check_dataset_exists():
                print("\n[WARNING] Dataset belum ada! Menjalankan generator otomatis dahulu...")
                from src.generator import create_dataset
                create_dataset()
                
            from src.benchmark import run_benchmark
            print("\n--- Memulai Benchmark Kecepatan 5 Model ---")
            run_benchmark()
            
        elif choice == '6':
            print("\nTerima kasih telah menggunakan Batik Motif Analysis CLI! Sampai jumpa.")
            break
        else:
            print("\nPilihan tidak valid. Silakan pilih nomor 1 hingga 6.")
            
        input("\nTekan [Enter] untuk kembali ke menu utama...")

if __name__ == "__main__":
    main()
