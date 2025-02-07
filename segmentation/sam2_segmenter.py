import os
import shutil
import cv2
import torch
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT).to(device)

# Crea el generador automático de máscaras
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    pred_iou_thresh=0.85,
    stability_score_thresh=0.85
    # Otros parámetros opcionales:
    # points_per_batch=32,
    # min_mask_region_area=100,
    # ...
)

# Carga de la imagen
INPUT_IMAGE_PATH = "input_image.jpg"
image = cv2.imread(INPUT_IMAGE_PATH)
if image is None:
    raise FileNotFoundError(f"No se pudo cargar la imagen en {INPUT_IMAGE_PATH}")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Genera las máscaras automáticamente
masks = mask_generator.generate(image_rgb)

# Crear/limpiar carpeta de salida
OUTPUT_DIR = "output_segments"
# Si la carpeta existe, la borramos para que empiece limpia
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# masks es una lista de dicts, cada dict contiene entre otras cosas:
# - 'segmentation': la máscara binaria
# - 'area': área de la máscara
# - 'bbox': bounding box [x, y, ancho, alto]
# - 'predicted_iou': IOU predicho
# - 'point_coords': puntos internos que determinaron la máscara
# - 'stability_score': puntaje de estabilidad
for i, mask_data in enumerate(masks):
    # Imprimir información de la máscara
    print(f"--- Segmento {i} ---")
    print(f"Área: {mask_data.get('area')}")
    print(f"BBox: {mask_data.get('bbox')}")
    print(f"IoU predicho: {mask_data.get('predicted_iou')}")
    print(f"Stability Score: {mask_data.get('stability_score')}")
    print(f"Coordenadas de punto: {mask_data.get('point_coords')}")
    print("-------------------")

    # Extraer la máscara y generar una imagen segmentada
    segmentation = mask_data["segmentation"].astype(np.uint8)
    segment = cv2.bitwise_and(image, image, mask=segmentation * 255)

    # Guardar la imagen resultante
    output_path = os.path.join(OUTPUT_DIR, f"automatic_segment_{i}.png")
    cv2.imwrite(output_path, segment)
    print(f"Guardado: {output_path}\n")

print("Segmentación automática completada.")
