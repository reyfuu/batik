import os
import cv2
import numpy as np

def generate_parang(width=128, height=128):
    # Latar krem (BGR)
    img = np.full((height, width, 3), (170, 200, 220), dtype=np.uint8)

    # Garis diagonal berulang khas Parang
    spacing = 24
    for offset in range(-height, width, spacing):
        cv2.line(img, (offset, 0), (offset + height, height), (20, 40, 60), 4)      # cokelat tua
        cv2.line(img, (offset + 12, 0), (offset + 12 + height, height), (80, 140, 180), 2)  # emas tipis
        # Titik isen-isen sepanjang diagonal
        for t in range(0, height, 16):
            cv2.circle(img, (offset + t + 6, t + 6), 1, (20, 40, 60), -1)

    # Noise ringan meniru tekstur kain
    noise = np.random.normal(0, 5, img.shape).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img

def generate_kawung(width=128, height=128):
    # Latar cokelat tua (BGR)
    img = np.full((height, width, 3), (10, 25, 40), dtype=np.uint8)

    grid_size = 32
    r1 = 18  # radius luar
    r2 = 8   # radius dalam

    for y in range(0, height + grid_size, grid_size):
        for x in range(0, width + grid_size, grid_size):
            cv2.circle(img, (x, y), r1, (170, 200, 220), 1)   # lingkaran krem luar
            cv2.circle(img, (x, y), r2, (80, 140, 180), -1)   # lingkaran emas dalam
            # Titik di perpotongan tengah
            cv2.circle(img, (x + grid_size // 2, y + grid_size // 2), 3, (170, 200, 220), -1)
            cv2.circle(img, (x + grid_size // 2, y + grid_size // 2), 1, (10, 25, 40), -1)

    # Noise tekstur
    noise = np.random.normal(0, 5, img.shape).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img

def generate_megamendung(width=128, height=128):
    # Latar indigo pekat (BGR)
    img = np.full((height, width, 3), (120, 40, 10), dtype=np.uint8)

    # Gelombang awan horizontal khas Megamendung
    for y in range(16, height, 32):
        # Tepian awan berlapis, gradasi biru muda -> cyan
        for layer in range(4):
            color_val = 140 + layer * 30
            color = (color_val, int(color_val * 0.8), int(color_val * 0.3))

            # Polyline bergelombang
            pts = []
            for x in range(0, width + 10, 10):
                wave = np.sin((x + y * 1.5) * 0.08) * 8
                pts.append([x, int(y + wave + (3 - layer) * 3)])
            pts = np.array(pts, np.int32)
            cv2.polylines(img, [pts], False, color, 2 + layer)

            # Bulatan awan
            for cx in range(16, width, 48):
                cv2.circle(img, (cx + (y % 4) * 4, y), 8 - layer, color, -1)

    # Noise tekstur
    noise = np.random.normal(0, 5, img.shape).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img

def create_dataset(base_dir="data", num_train=40, num_test=15):
    """Buat dataset sintetis Parang, Kawung, Megamendung."""
    categories = ["parang", "kawung", "megamendung"]
    generators = {
        "parang": generate_parang,
        "kawung": generate_kawung,
        "megamendung": generate_megamendung
    }
    
    print(f"Membuat dataset batik sintetis di folder: {base_dir}...")
    for split, count in [("train", num_train), ("test", num_test)]:
        for cat in categories:
            cat_dir = os.path.join(base_dir, split, cat)
            os.makedirs(cat_dir, exist_ok=True)
            
            for i in range(count):
                img = generators[cat](width=128, height=128)
                filepath = os.path.join(cat_dir, f"{cat}_{i:03d}.png")
                cv2.imwrite(filepath, img)
                
    print(f"Dataset berhasil dibuat: {num_train} data latih & {num_test} data uji per kelas.")

if __name__ == "__main__":
    create_dataset()
