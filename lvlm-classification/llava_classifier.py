import os
import ollama
import time

# Carpeta donde está este script:
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construimos la ruta a output_segments a partir de la carpeta del script
SEGMENT_DIR = os.path.join(script_dir, "..", "segmentation", "output_segments")

DESC_DIR = os.path.join(script_dir, "output_descriptions")

os.makedirs(DESC_DIR, exist_ok=True)

timestamp = time.strftime("%Y-%m-%d_%H:%M:%S")
txt_filename = f"descriptions_{timestamp}.txt"
txt_path = os.path.join(DESC_DIR, txt_filename)

segment_images = [f for f in os.listdir(SEGMENT_DIR) if f.endswith(".png")]
if not segment_images:
    raise FileNotFoundError(f"No se encontraron imágenes PNG en {SEGMENT_DIR}")

with open(txt_path, "w", encoding="utf-8") as f:
    for i, seg_file in enumerate(segment_images):
        segment_path = os.path.join(SEGMENT_DIR, seg_file)
        res = ollama.chat(
            model="llava:34b",
            messages=[
                {
                    "role": "user",
                    "content": "Describe the image.",
                    "images": [segment_path]
                }
            ]
        )

        description = res["message"]["content"]
        output_line = f"Segmento {i} ({seg_file}): {description}"
        print(output_line)
        f.write(output_line + "\n")

print(f"Descripciones guardadas en {txt_path}")
