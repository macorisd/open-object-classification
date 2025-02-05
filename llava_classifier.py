import os
import cv2
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image

MODEL_NAME = "llava-hf/llava-1.5-7b-hf"
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = LlavaForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)

SEGMENT_DIR = "output_segments"
OUTPUT_FILE = "classifications.txt"

segment_images = [f for f in os.listdir(SEGMENT_DIR) if f.endswith(".png")]

if not segment_images:
    raise FileNotFoundError(f"No se encontraron segmentos en {SEGMENT_DIR}")

results = []
for i, segment_file in enumerate(segment_images):
    image_path = os.path.join(SEGMENT_DIR, segment_file)
    image = Image.open(image_path).convert("RGB")

    inputs = processor(images=image, text="¿Qué objeto ves en esta imagen? Responde con una sola palabra, a modo de categoría semántica.", return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(**inputs)
        classification = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    results.append(f"{segment_file}: {classification}")
    print(f"Segmento {i}: {classification}")

with open(OUTPUT_FILE, "w") as f:
    f.write("\n".join(results))

print(f"Clasificaciones guardadas en {OUTPUT_FILE}.")
