import os
import ollama
import time

script_dir = os.path.dirname(os.path.abspath(__file__))

# Carpeta donde se encuentra el .txt con las descripciones (salida de LLaVA)
DESCRIPTIONS_DIR = os.path.join(script_dir, "..", "lvlm-description", "output_descriptions")

# Carpeta para guardar la clasificación resultante
CLASSIFICATION_DIR = os.path.join(script_dir, "output_descriptions")
os.makedirs(CLASSIFICATION_DIR, exist_ok=True)

def main():
    # 1. Buscar el único archivo .txt en DESCRIPTIONS_DIR
    txt_files = [f for f in os.listdir(DESCRIPTIONS_DIR) if f.endswith(".txt")]
    if not txt_files:
        raise FileNotFoundError(f"No se encontraron archivos .txt en {DESCRIPTIONS_DIR}")
    if len(txt_files) > 1:
        print("ADVERTENCIA: Se encontraron múltiples archivos .txt. Se usará el primero.")
    txt_file = txt_files[0]
    txt_path = os.path.join(DESCRIPTIONS_DIR, txt_file)

    # 2. Leer su contenido
    with open(txt_path, "r", encoding="utf-8") as f:
        descriptions_content = f.read()

    # 3. Preparar el prompt y llamar a 'deepseek' vía Ollama
    # Ajusta 'model' según el modelo que tengas disponible
    prompt = f"Te voy a pasar una lista de descripciones de segmentos de imágenes. Los segmentos han sido extraídos con SAM2 y la descripción de las imágenes ha sido realizada con LLaVA. Te voy a pasar la descripción de cada segmento, y tu tarea es decirme únicamente los segmentos que crees que tienen objetos en ellos. Habrá algunos confusos o difusos, esos descártalos. Dime sólo qué segmentos tienen, con total seguridad, alguna detección exitosa: {descriptions_content}"
    response = ollama.chat(
        model="deepseek-r1:14b",  # <--- Cambia aquí si tu modelo tiene otro nombre, e.g. "deepseek:13b"
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    # 4. Guardar la respuesta en un archivo con timestamp
    deepseek_answer = response["message"]["content"]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_filename = f"classification_{timestamp}.txt"
    output_path = os.path.join(CLASSIFICATION_DIR, output_filename)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(deepseek_answer)

    print(f"Respuesta de Deepseek guardada en {output_path}")
    print("Contenido:\n", deepseek_answer)


if __name__ == "__main__":
    main()
