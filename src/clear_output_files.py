import os

def clear_output_segments(directory: str) -> None:
    """
    Deletes all .png files in the 'segmentation/output_segments' directory.
    """
    for filename in os.listdir(os.path.join(directory, "segmentation", "output_segments")):
        if filename.endswith(".png"):
            file_path = os.path.join(directory, "segmentation", "output_segments", filename)
            os.remove(file_path)
            print(f"Deleted {file_path}")

def clear_output_descriptions(directory: str) -> None:
    """
    Deletes all .txt files in the 'lvlm-description/output_descriptions' directory.
    """
    for filename in os.listdir(os.path.join(directory, "lvlm-description", "output_descriptions")):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, "lvlm-description", "output_descriptions", filename)
            os.remove(file_path)
            print(f"Deleted {file_path}")

def clear_output_classifications(directory: str) -> None:
    """
    Deletes all .txt files in the 'llm-classification/output_classification' directory.
    """
    for filename in os.listdir(os.path.join(directory, "llm-classification", "output_classification")):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, "llm-classification", "output_classification", filename)
            os.remove(file_path)
            print(f"Deleted {file_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    clear_output_segments(script_dir)
    clear_output_descriptions(script_dir)
    clear_output_classifications(script_dir)