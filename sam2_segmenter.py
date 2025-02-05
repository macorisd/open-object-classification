import cv2
import torch
import numpy as np
import os
from segment_anything import SamPredictor, sam_model_registry

SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT).to(device)
predictor = SamPredictor(sam)

INPUT_IMAGE_PATH = "input_image2.jpg"
image = cv2.imread(INPUT_IMAGE_PATH)
if image is None:
    raise FileNotFoundError(f"No se pudo cargar la imagen en {INPUT_IMAGE_PATH}")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(image_rgb)

masks, _, _ = predictor.predict(point_coords=None, point_labels=None)

OUTPUT_DIR = "output_segments"
os.makedirs(OUTPUT_DIR, exist_ok=True)

for i, mask in enumerate(masks):
    segment = cv2.bitwise_and(image, image, mask=mask.astype(np.uint8) * 255)
    segment_path = os.path.join(OUTPUT_DIR, f"segment_{i}.png")
    cv2.imwrite(segment_path, segment)
    print(f"Guardado: {segment_path}")

print("Segmentación completada. Las imágenes están en la carpeta 'output_segments'.")
