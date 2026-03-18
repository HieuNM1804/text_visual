from pathlib import Path
import cv2
import numpy as np

root = Path(r"D:\Research\VLM_project\dataset\QuickDraw\sketch")  # folder chứa các folder nhãn
kernel = np.ones((2, 2), np.uint8)             # (2,2) nhẹ | (3,3) vừa

for img_path in root.rglob("*.png"):            # đổi *.png nếu cần
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue

    _, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
    bold = cv2.dilate(binary, kernel, iterations=1)
    bold = cv2.bitwise_not(bold)

    cv2.imwrite(str(img_path), bold)             # 🔥 ghi đè ảnh gốc
